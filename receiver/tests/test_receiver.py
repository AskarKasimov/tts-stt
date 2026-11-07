import asyncio
import copy
import os

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from speech_receiver.config import Config, load
from speech_receiver.engine import Engine, Rejected
from speech_receiver.http import install
from speech_receiver.journal import KafkaJournal

HIGH = "server-secret-for-tests-00000001"
NORMAL = "developer-secret-for-tests-00001"


class MemoryJournal:
    def __init__(self, *_):
        self.records = {}
        self.fail_result = False

    def replay(self):
        return copy.deepcopy(self.records)

    def append(self, record, delete=False):
        if self.fail_result and record.get("state") == "succeeded":
            raise RuntimeError("unconfirmed")
        if delete:
            self.records.pop(record["id"], None)
        else:
            self.records[record["id"]] = copy.deepcopy(record)

    def close(self):
        pass


def config(path, **kwargs):
    return Config(
        operation="tts",
        data_dir=str(path),
        brokers=["unused:9092"],
        topic="test",
        clients={
            "server": {"token": HIGH, "priority": "high"},
            "dev": {"token": NORMAL, "priority": "normal"},
        },
        instances=[],
        **kwargs,
    )


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


def test_priority_idempotency_cancellation_and_restart(tmp_path):
    async def scenario():
        journal = MemoryJournal()
        gate = asyncio.Event()
        calls = []

        async def execute(body, *_):
            calls.append(body)
            if body == b"A":
                await gate.wait()
            return 200, "audio/wav", b"wav:" + body

        engine = Engine(config(tmp_path), journal, execute)
        await engine.start()
        try:
            await engine.submit("dev", "normal", "A", b"A", "text/plain", b"")
            await until(lambda: calls == [b"A"])
            b = await engine.submit("dev", "normal", "B", b"B", "text/plain", b"")
            c = await engine.submit("server", "high", "C", b"C", "text/plain", b"")
            cancelled = await engine.submit(
                "dev", "normal", "cancel", b"cancel", "text/plain", b""
            )
            await engine.detach(cancelled)
            gate.set()
            assert (await engine.response(b))[2] == b"wav:B"
            assert calls == [b"A", b"C", b"B"]
            assert (
                await engine.submit("dev", "normal", "B", b"B", "text/plain", b"") == b
            )
            with pytest.raises(Rejected) as conflict:
                await engine.submit(
                    "dev", "normal", "B", b"different", "text/plain", b""
                )
            assert conflict.value.status == 409
        finally:
            gate.set()
            await engine.close()
        restored = Engine(config(tmp_path), journal, execute)
        await restored.start()
        try:
            assert (await restored.response(c))[2] == b"wav:C"
            assert calls == [b"A", b"C", b"B"]
        finally:
            await restored.close()

    asyncio.run(scenario())


def test_normal_limit_reserves_high_admission(tmp_path):
    async def scenario():
        gate = asyncio.Event()

        async def execute(*_):
            await gate.wait()
            return 200, "audio/wav", b"wav"

        engine = Engine(config(tmp_path, max_normal=1), MemoryJournal(), execute)
        await engine.start()
        try:
            await engine.submit("dev", "normal", "running", b"1", "x", b"")
            await until(
                lambda: any(j["state"] == "running" for j in engine.jobs.values())
            )
            await engine.submit("dev", "normal", "waiting", b"2", "x", b"")
            with pytest.raises(Rejected) as full:
                await engine.submit("dev", "normal", "full", b"3", "x", b"")
            assert full.value.status == 429
            await engine.submit("server", "high", "important", b"4", "x", b"")
        finally:
            gate.set()
            await engine.close()

    asyncio.run(scenario())


def test_timeout_does_not_release_running_model(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        calls = []

        async def execute(body, *_):
            calls.append(body)
            await gate.wait()
            return 200, "audio/wav", b"wav"

        engine = Engine(
            config(tmp_path, execution_timeout=0.05), MemoryJournal(), execute
        )
        await engine.start()
        try:
            a = await engine.submit("dev", "normal", "A", b"A", "x", b"")
            await until(lambda: bool(calls))
            with pytest.raises(Rejected) as timeout:
                await engine.response(a)
            assert timeout.value.execution == "uncertain"
            await engine.detach(a)
            await engine.submit("server", "high", "B", b"B", "x", b"")
            await asyncio.sleep(0.1)
            assert calls == [b"A"]
        finally:
            gate.set()
            await engine.close()

    asyncio.run(scenario())


def test_result_write_failure_never_returns_uncommitted_success(tmp_path):
    async def scenario():
        journal = MemoryJournal()
        journal.fail_result = True

        async def execute(*_):
            return 200, "audio/wav", b"wav"

        engine = Engine(config(tmp_path), journal, execute)
        await engine.start()
        try:
            a = await engine.submit("dev", "normal", "A", b"A", "x", b"")
            await until(lambda: engine.fatal.is_set())
            assert not engine.ready
            assert engine.jobs[a]["state"] == "running"
            assert journal.records[a]["state"] == "running"
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_http_auth_and_round_trip(tmp_path):
    async def scenario():
        app = FastAPI()

        @app.post("/synthesize")
        async def synthesize(data: dict):
            return Response(data["text"].encode(), media_type="audio/wav")

        install(app, config(tmp_path), lambda data: data["text"], MemoryJournal)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                assert (
                    await client.post("/synthesize", json={"text": "hello"})
                ).status_code == 401
                response = await client.post(
                    "/synthesize",
                    json={"text": "hello"},
                    headers={
                        "Authorization": "Bearer " + NORMAL,
                        "Idempotency-Key": "same",
                    },
                )
                assert response.status_code == 200
                assert response.content == b"hello"
                assert response.headers["x-voice-execution-state"] == "finished"
                assert (await client.get("/health/ready")).status_code == 200

    asyncio.run(scenario())


def test_config_rejects_duplicates_unknown_fields_and_tokens(tmp_path):
    path = tmp_path / "prod.yaml"
    path.write_text("operation: stt\noperation: tts\n")
    with pytest.raises(ValueError):
        load(path)
    with pytest.raises((TypeError, ValueError)):
        config(tmp_path, mode="unknown")
    with pytest.raises(ValueError):
        value = config(tmp_path)
        value.clients["dev"]["token"] = HIGH
        value.__post_init__()


def test_volume_cannot_be_owned_twice(tmp_path):
    from speech_receiver.engine import Blobs

    first = Blobs(tmp_path, 1024)
    try:
        with pytest.raises(RuntimeError):
            Blobs(tmp_path, 1024)
    finally:
        first.close()


def test_redpanda_durable_round_trip(tmp_path):
    brokers = os.getenv("TEST_KAFKA_BROKERS")
    if not brokers:
        pytest.skip("TEST_KAFKA_BROKERS is not set")
    import uuid

    from confluent_kafka.admin import AdminClient

    topic = "receiver-test-" + uuid.uuid4().hex

    async def scenario():
        cfg = config(tmp_path)
        cfg.brokers, cfg.topic = brokers.split(","), topic
        calls = []

        async def execute(body, *_):
            calls.append(body)
            return 200, "audio/wav", b"result"

        engine = Engine(cfg, KafkaJournal(cfg.brokers, topic), execute)
        await engine.start()
        try:
            job = await engine.submit(
                "dev", "normal", "test", b"audio", "audio/wav", b""
            )
            assert (await engine.response(job))[2] == b"result"
        finally:
            await engine.close()
        engine = Engine(cfg, KafkaJournal(cfg.brokers, topic), execute)
        await engine.start()
        try:
            assert (await engine.response(job))[2] == b"result"
            assert calls == [b"audio"]
        finally:
            await engine.close()

    try:
        asyncio.run(scenario())
    finally:
        admin = AdminClient(
            {"bootstrap.servers": brokers, "broker.address.family": "v4"}
        )
        admin.delete_topics([topic])[topic].result(15)


def test_recovery_cancels_waiting_and_holds_interrupted_slots(tmp_path):
    async def scenario():
        journal = MemoryJournal()

        async def execute(*_):
            raise AssertionError("interrupted jobs must not restart")

        engine = Engine(config(tmp_path), journal, execute)
        await engine.start(dispatch=False)
        a = await engine.submit("dev", "normal", "A", b"A", "x", b"")
        b = await engine.submit("server", "high", "B", b"B", "x", b"")
        engine.jobs[b].update(state="running", instance="local", started=1)
        await engine.persist(engine.jobs[b])
        await engine.close()
        engine = Engine(config(tmp_path), journal, execute)
        await engine.start()
        try:
            assert engine.jobs[a]["state"] == "failed"
            assert engine.jobs[a]["execution"] == "not-started"
            assert engine.jobs[b]["held"]
            await engine.release_held("local")
            assert not engine.jobs[b]["held"]
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_broker_fences_second_topic_owner():
    brokers = os.getenv("TEST_KAFKA_BROKERS")
    if not brokers:
        pytest.skip("TEST_KAFKA_BROKERS is not set")
    import uuid

    from confluent_kafka.admin import AdminClient

    topic = "receiver-fence-" + uuid.uuid4().hex
    first = KafkaJournal(brokers.split(","), topic)
    second = None
    try:
        first.append({"id": "job", "state": "queued"})
        second = KafkaJournal(brokers.split(","), topic)
        assert second.replay()["job"]["state"] == "queued"
        with pytest.raises(Exception):
            first.append({"id": "job", "state": "running"})
    finally:
        first.close()
        if second:
            second.close()
        admin = AdminClient(
            {"bootstrap.servers": brokers, "broker.address.family": "v4"}
        )
        admin.delete_topics([topic])[topic].result(15)


def test_remote_instances_and_held_transport_failure(tmp_path):
    async def scenario():
        cfg = config(tmp_path)
        cfg.instances = [
            {
                "id": "remote-a",
                "url": "http://model-a/synthesize",
                "token": HIGH,
                "slots": 1,
            },
            {
                "id": "remote-b",
                "url": "http://model-b/synthesize",
                "token": HIGH,
                "slots": 1,
            },
        ]
        cfg.__post_init__()
        engine = Engine(cfg, MemoryJournal(), lambda *_: None)
        gate = asyncio.Event()
        calls = []

        async def model(request):
            assert request.headers["authorization"] == "Bearer " + HIGH
            calls.append(request.url.host)
            if request.url.host == "model-a":
                raise httpx.ReadError("connection interrupted")
            await gate.wait()
            return httpx.Response(
                200,
                content=b"wav",
                headers={
                    "content-type": "audio/wav",
                    "x-voice-execution-state": "finished",
                },
            )

        await engine.http.aclose()
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(model))
        await engine.start()
        try:
            a = await engine.submit("dev", "normal", "A", b"A", "x", b"")
            b = await engine.submit("server", "high", "B", b"B", "x", b"")
            await until(
                lambda: engine.jobs[a].get("held") or engine.jobs[b].get("held")
            )
            assert "model-a" in calls and "model-b" in calls
            held = [j for j in engine.jobs.values() if j.get("held")]
            assert len(held) == 1 and held[0]["instance"] == "remote-a"
        finally:
            gate.set()
            await engine.close()

    asyncio.run(scenario())


def test_http_priority_admission_is_independent(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        calls = []
        app = FastAPI()

        @app.post("/synthesize")
        async def synthesize(data: dict):
            calls.append(data["text"])
            await gate.wait()
            return Response(b"wav", media_type="audio/wav")

        install(
            app,
            config(tmp_path, max_normal=1),
            lambda data: data["text"],
            MemoryJournal,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                normal = {"Authorization": "Bearer " + NORMAL}
                first = asyncio.create_task(
                    client.post("/synthesize", json={"text": "A"}, headers=normal)
                )
                await until(lambda: bool(calls))
                second = asyncio.create_task(
                    client.post("/synthesize", json={"text": "B"}, headers=normal)
                )
                await until(lambda: len(app.state.receiver.jobs) == 2)
                assert (
                    await client.post("/synthesize", json={"text": "C"}, headers=normal)
                ).status_code == 429
                high = asyncio.create_task(
                    client.post(
                        "/synthesize",
                        json={"text": "H"},
                        headers={"Authorization": "Bearer " + HIGH},
                    )
                )
                await until(lambda: len(app.state.receiver.jobs) == 3)
                gate.set()
                await asyncio.gather(first, second, high)
                assert calls == ["A", "H", "B"]

    asyncio.run(scenario())


def test_volume_lock_precedes_producer_fencing(tmp_path):
    async def scenario():
        created = []

        def journal():
            created.append(True)
            return MemoryJournal()

        async def execute(*_):
            return 200, "audio/wav", b"wav"

        first = Engine(config(tmp_path), journal, execute)
        try:
            with pytest.raises(RuntimeError):
                Engine(config(tmp_path), journal, execute)
            assert len(created) == 1
        finally:
            await first.close()

    asyncio.run(scenario())


def test_cleanup_failure_keeps_committed_result(tmp_path):
    async def scenario():
        journal = MemoryJournal()

        async def execute(*_):
            return 200, "audio/wav", b"committed"

        engine = Engine(config(tmp_path), journal, execute)
        await engine.start()
        original = engine.blobs.remove

        def failed_cleanup(_):
            raise OSError("disk failure")

        engine.blobs.remove = failed_cleanup
        try:
            job = await engine.submit("dev", "normal", "A", b"A", "x", b"")
            await until(lambda: engine.fatal.is_set())
            assert (await engine.response(job))[2] == b"committed"
            assert journal.records[job]["state"] == "succeeded"
        finally:
            engine.blobs.remove = original
            await engine.close()

    asyncio.run(scenario())


def test_duplicate_receiver_rejected_before_model_load(tmp_path):
    from contextlib import asynccontextmanager

    async def scenario():
        loads = []

        @asynccontextmanager
        async def model_lifespan(_):
            loads.append(True)
            yield

        first = install(
            FastAPI(lifespan=model_lifespan),
            config(tmp_path),
            lambda _: None,
            MemoryJournal,
        )
        second = install(
            FastAPI(lifespan=model_lifespan),
            config(tmp_path),
            lambda _: None,
            MemoryJournal,
        )
        async with first.router.lifespan_context(first):
            with pytest.raises(RuntimeError):
                async with second.router.lifespan_context(second):
                    pass
            assert len(loads) == 1

    asyncio.run(scenario())


def test_http_cancel_during_durable_admission_never_starts_model(tmp_path):
    import threading

    async def scenario():
        entered = threading.Event()
        release = threading.Event()

        class SlowJournal(MemoryJournal):
            def append(self, record, delete=False):
                if record["state"] == "queued":
                    entered.set()
                    assert release.wait(5)
                super().append(record, delete)

        app = FastAPI()
        calls = []

        @app.post("/synthesize")
        async def synthesize(data: dict):
            calls.append(data)
            return Response(b"wav", media_type="audio/wav")

        install(app, config(tmp_path), lambda _: None, SlowJournal)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                task = asyncio.create_task(
                    client.post(
                        "/synthesize",
                        json={"text": "hello"},
                        headers={"Authorization": "Bearer " + NORMAL},
                    )
                )
                try:
                    await until(entered.is_set)
                    task.cancel()
                    await asyncio.sleep(0)
                finally:
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert calls == []
                assert all(
                    j["state"] == "failed" and j["reason"] == "cancelled"
                    for j in app.state.receiver.jobs.values()
                )

    asyncio.run(scenario())


def test_retention_removes_finished_records_and_payloads(tmp_path):
    async def scenario():
        journal = MemoryJournal()

        async def execute(*_):
            return 200, "audio/wav", b"wav"

        engine = Engine(config(tmp_path, retention=0.01), journal, execute)
        await engine.start()
        try:
            job = await engine.submit("dev", "normal", "A", b"A", "x", b"")
            await engine.response(job)
            await engine.detach(job)
            await until(lambda: job not in engine.jobs)
            assert job not in journal.records
            assert not list(tmp_path.glob("*.blob"))
            with pytest.raises(Rejected) as expired:
                await engine.response(job)
            assert expired.value.status == 404
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_operation_mismatch_never_replays_another_service_queue(tmp_path):
    async def scenario():
        journal = MemoryJournal()

        async def execute(*_):
            raise AssertionError("wrong operation must not execute")

        first_config = config(tmp_path)
        first_config.operation = "stt"
        first = Engine(first_config, journal, execute)
        await first.start(dispatch=False)
        job = await first.submit("dev", "normal", "same", b"audio", "audio/wav", b"")
        await first.close()
        second = Engine(config(tmp_path), journal, execute)
        with pytest.raises(RuntimeError):
            await second.start()
        assert journal.records[job]["state"] == "queued"

    asyncio.run(scenario())
