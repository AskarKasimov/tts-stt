"""One compacted state topic per operation; payloads never enter Kafka."""

import hashlib
import json
import time
import uuid

from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, ConfigResource, NewTopic


class KafkaJournal:
    def __init__(self, brokers, topic):
        self.topic = topic
        base = {"bootstrap.servers": ",".join(brokers), "broker.address.family": "v4"}
        admin = AdminClient(base)
        metadata = admin.list_topics(timeout=10)
        if topic not in metadata.topics:
            try:
                admin.create_topics(
                    [
                        NewTopic(
                            topic,
                            num_partitions=1,
                            replication_factor=1,
                            config={"cleanup.policy": "compact", "retention.ms": "-1"},
                        )
                    ]
                )[topic].result(15)
            except Exception:
                # A concurrent topic creator is harmless; verify the actual topic below.
                pass
        metadata = admin.list_topics(topic, timeout=10).topics[topic]
        if metadata.error or len(metadata.partitions) != 1:
            raise RuntimeError("receiver topic must have exactly one partition")
        resource = ConfigResource(ConfigResource.Type.TOPIC, topic)
        settings = admin.describe_configs([resource])[resource].result(15)
        if (
            settings["cleanup.policy"].value != "compact"
            or settings["retention.ms"].value != "-1"
        ):
            raise RuntimeError(
                "receiver topic requires compaction and unlimited retention"
            )
        self.producer = Producer(
            {
                **base,
                "enable.idempotence": True,
                "acks": "all",
                "message.timeout.ms": 10000,
                "transactional.id": "speech-receiver-"
                + hashlib.sha256(topic.encode()).hexdigest(),
                "transaction.timeout.ms": 15000,
            }
        )
        self.producer.init_transactions(20)
        self.consumer = Consumer(
            {
                **base,
                "group.id": "speech-restore-" + uuid.uuid4().hex,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
                "enable.partition.eof": True,
                "isolation.level": "read_committed",
            }
        )

    def replay(self):
        partition = TopicPartition(self.topic, 0)
        low, high = self.consumer.get_watermark_offsets(
            partition, timeout=10, cached=False
        )
        self.consumer.assign([TopicPartition(self.topic, 0, low)])
        records = {}
        position = low
        deadline = time.monotonic() + 30
        while position < high:
            if time.monotonic() >= deadline:
                raise RuntimeError("receiver restore barrier timed out")
            message = self.consumer.poll(1)
            if message is None:
                continue
            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    position = max(position, message.offset())
                    continue
                raise RuntimeError("receiver restore failed")
            position = message.offset() + 1
            key = message.key().decode()
            if message.value() is None:
                records.pop(key, None)
            else:
                record = json.loads(message.value())
                if record["id"] != key:
                    raise RuntimeError("invalid receiver record")
                records[key] = record
        return records

    def append(self, record, delete=False):
        result = []
        self.producer.begin_transaction()
        self.producer.produce(
            self.topic,
            partition=0,
            key=record["id"],
            value=None
            if delete
            else json.dumps(record, separators=(",", ":")).encode(),
            on_delivery=lambda error, _: result.append(error),
        )
        if self.producer.flush(12) or not result or result[0] is not None:
            raise RuntimeError("receiver durable write unconfirmed")
        self.producer.commit_transaction(15)

    def close(self):
        self.consumer.close()
        self.producer.flush(12)
