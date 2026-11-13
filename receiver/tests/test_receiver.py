import asyncio
import copy
import os
from pathlib import Path

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


@pytest.mark.parametrize("service", ["gigaam", "voxcpm2"])
def test_committed_clients_assign_production_high_and_development_normal(service):
    root = Path(__file__).resolve().parents[2]
    cfg = load(root / service / "config/prod.yaml")
    other = "voxcpm2" if service == "gigaam" else "gigaam"
    other_cfg = load(root / other / "config/prod.yaml")
    clients_match = cfg.clients == other_cfg.clients
    assert clients_match, "STT and TTS must register the same clients"
    tokens_differ = cfg.clients["production"]["token"] != cfg.clients["development"]["token"]
    assert tokens_differ, "Production and development must use distinct tokens"
    for client, priority in (("production", "high"), ("development", "normal")):
        identity = cfg.authenticate("Bearer " + cfg.clients[client]["token"])
        assert identity == (client, priority)
    unknown_identity = cfg.authenticate("Bearer unknown-client")
    assert unknown_identity is None


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
                lambda: (
                    (engine.jobs[a].get("held") or engine.jobs[b].get("held"))
                    and "model-b" in calls
                )
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
            await until(lambda: not list(tmp_path.glob("*.blob")))
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
        job = await first.submit(
            "dev", "normal", "same", b"audio", "audio/wav", b"model=v3_ctc"
        )
        await first.close()
        second = Engine(config(tmp_path), journal, execute)
        with pytest.raises(RuntimeError):
            await second.start()
        assert journal.records[job]["state"] == "queued"

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["put", "get", "remove"])
def test_slow_blob_io_keeps_health_metrics_and_auth_responsive(
    tmp_path, monkeypatch, method
):
    import threading

    async def scenario():
        entered, release = threading.Event(), threading.Event()
        app = FastAPI()

        @app.post("/synthesize")
        async def synthesize(data: dict):
            return Response(b"wav", media_type="audio/wav")

        @app.get("/metrics")
        async def metrics():
            return Response(b"# model metrics\n", media_type="text/plain")

        install(app, config(tmp_path), lambda _: None, MemoryJournal)
        async with app.router.lifespan_context(app):
            engine = app.state.receiver
            original = getattr(engine.blobs, method)

            def slow(*args):
                entered.set()
                assert release.wait(3), "blob operation blocked the event loop"
                return original(*args)

            monkeypatch.setattr(engine.blobs, method, slow)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                request = asyncio.create_task(
                    client.post(
                        "/synthesize",
                        json={"text": "hello"},
                        headers={"Authorization": "Bearer " + NORMAL},
                    )
                )
                try:
                    await until(entered.is_set)
                    async with asyncio.timeout(0.5):
                        assert (await client.get("/health/ready")).status_code == 200
                        assert "receiver_ready 1" in (await client.get("/metrics")).text
                        assert (
                            await client.post("/synthesize", json={})
                        ).status_code == 401
                finally:
                    release.set()
                    assert (await request).status_code == 200

    asyncio.run(scenario())


def test_blob_cancellation_drains_writer_and_reserved_quota_stays_bounded(
    tmp_path, monkeypatch
):
    import threading

    async def scenario():
        entered, release = threading.Event(), threading.Event()
        engine = Engine(
            config(tmp_path, max_payload_bytes=4), MemoryJournal(), lambda *_: None
        )
        await engine.start(dispatch=False)
        original = engine.blobs.sync

        def slow_sync():
            entered.set()
            assert release.wait(3)
            return original()

        monkeypatch.setattr(engine.blobs, "sync", slow_sync)
        first = asyncio.create_task(engine.store_payload(b"full"))
        try:
            await until(entered.is_set)
            first.cancel()
            async with asyncio.timeout(0.5):
                with pytest.raises(Rejected) as quota:
                    await engine.store_payload(b"x")
                assert quota.value.reason == "payload_quota"
            assert not first.done() and engine.blobs.used == 4
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert engine.blobs.used == 4
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)
            await engine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [429, 503])
def test_remote_not_started_retries_without_caching_temporary_refusal(tmp_path, status):
    async def scenario():
        cfg = config(tmp_path)
        cfg.instances = [
            {
                "id": "worker",
                "url": "http://worker/synthesize",
                "token": HIGH,
                "slots": 1,
            }
        ]
        journal = MemoryJournal()
        engine = Engine(cfg, journal, lambda *_: None)
        calls = []

        async def worker(_):
            calls.append(True)
            if len(calls) == 1:
                return httpx.Response(
                    status,
                    json={"error": "worker_busy"},
                    headers={"x-voice-execution-state": "not-started"},
                )
            return httpx.Response(
                200,
                content=b"wav",
                headers={
                    "x-voice-execution-state": "finished",
                    "content-type": "audio/wav",
                },
            )

        await engine.http.aclose()
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(worker))
        await engine.start()
        try:
            job = await engine.submit("dev", "normal", "same", b"body", "x", b"")
            await until(lambda: engine.jobs[job].get("retry_at"))
            assert engine.jobs[job]["state"] == "queued"
            assert not engine.jobs[job].get("held")
            assert journal.records[job]["execution"] == "not-started"
            assert (await engine.response(job))[2:] == (b"wav", "finished")
            assert (
                await engine.submit("dev", "normal", "same", b"body", "x", b"") == job
            )
            assert (await engine.response(job))[2:] == (b"wav", "finished")
            assert len(calls) == 2
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_remote_not_started_terminal_response_is_preserved_on_replay(tmp_path):
    async def scenario():
        cfg = config(tmp_path)
        cfg.instances = [
            {
                "id": "worker",
                "url": "http://worker/synthesize",
                "token": HIGH,
                "slots": 1,
            }
        ]
        journal = MemoryJournal()
        engine = Engine(cfg, journal, lambda *_: None)

        async def worker(_):
            return httpx.Response(
                422,
                content=b"invalid",
                headers={"x-voice-execution-state": "not-started"},
            )

        await engine.http.aclose()
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(worker))
        await engine.start()
        try:
            job = await engine.submit("dev", "normal", "same", b"body", "x", b"")
            response = await engine.response(job)
            assert response[0] == 422 and response[3] == "not-started"
        finally:
            await engine.close()
        restored = Engine(cfg, journal, lambda *_: None)
        await restored.start()
        try:
            assert await restored.response(job) == response
            assert not restored.jobs[job]["held"]
        finally:
            await restored.close()

    asyncio.run(scenario())


def test_busy_worker_exhausts_original_queue_deadline_without_held_slot(tmp_path):
    async def scenario():
        cfg = config(tmp_path, queue_timeout=0.1)
        cfg.instances = [
            {
                "id": "worker",
                "url": "http://worker/synthesize",
                "token": HIGH,
                "slots": 1,
            }
        ]
        engine = Engine(cfg, MemoryJournal(), lambda *_: None)

        async def worker(_):
            return httpx.Response(
                429, headers={"x-voice-execution-state": "not-started"}
            )

        await engine.http.aclose()
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(worker))
        await engine.start()
        try:
            job = await engine.submit("dev", "normal", "same", b"body", "x", b"")
            with pytest.raises(Rejected) as timeout:
                await engine.response(job)
            assert timeout.value.reason == "queue_timeout"
            assert timeout.value.execution == "not-started"
            assert not engine.jobs[job]["held"]
        finally:
            await engine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["journal", "payload", "worker"])
def test_failure_logs_safe_operation_stage_and_error_type(
    tmp_path, monkeypatch, caplog, failure
):
    import json
    import logging

    caplog.set_level(logging.ERROR)
    secret = "SECRET-request-text-and-token"

    async def scenario():
        journal = MemoryJournal()

        async def execute(*_):
            raise RuntimeError(secret)

        engine = Engine(config(tmp_path), journal, execute)
        await engine.start()
        try:
            if failure == "journal":

                def broken(*_):
                    raise ConnectionError(secret)

                monkeypatch.setattr(journal, "append", broken)
            elif failure == "payload":

                def broken(*_):
                    raise OSError(28, secret)

                monkeypatch.setattr(engine.blobs, "put", broken)
            if failure in ("journal", "payload"):
                with pytest.raises(Rejected):
                    await engine.submit("dev", "normal", "same", b"body", "x", b"")
            else:
                job = await engine.submit("dev", "normal", "same", b"body", "x", b"")
                with pytest.raises(Rejected):
                    await engine.response(job)
        finally:
            await engine.close()

    asyncio.run(scenario())
    records = [json.loads(record.message) for record in caplog.records]
    stage, error_type = {
        "journal": ("journal_append", "ConnectionError"),
        "payload": ("payload_write", "OSError"),
        "worker": ("execution", "RuntimeError"),
    }[failure]
    assert any(
        r["operation"] == "tts"
        and r["stage"] == stage
        and r["error_type"] == error_type
        for r in records
    )
    if failure == "payload":
        assert any(record.get("errno") == 28 for record in records)
    assert secret not in caplog.text


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("remaining", [0, 1])
def test_not_started_retry_cancels_only_when_all_waiters_left(
    tmp_path, status, remaining
):
    async def scenario():
        cfg = config(tmp_path)
        cfg.instances = [
            {
                "id": "worker",
                "url": "http://worker/synthesize",
                "token": HIGH,
                "slots": 1,
            }
        ]
        journal = MemoryJournal()
        engine = Engine(cfg, journal, lambda *_: None)
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def worker(_):
            calls.append(True)
            if len(calls) == 1:
                entered.set()
                await release.wait()
                return httpx.Response(
                    status, headers={"x-voice-execution-state": "not-started"}
                )
            return httpx.Response(
                200, content=b"wav", headers={"x-voice-execution-state": "finished"}
            )

        await engine.http.aclose()
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(worker))
        await engine.start()
        try:
            job = await engine.submit("dev", "normal", "same", b"body", "x", b"")
            await entered.wait()
            if remaining:
                assert (
                    await engine.submit("dev", "normal", "same", b"body", "x", b"")
                    == job
                )
            await engine.detach(job)
            assert engine.jobs[job]["state"] == "running"
            release.set()
            if remaining:
                assert (await engine.response(job))[2] == b"wav"
                assert len(calls) == 2
            else:
                with pytest.raises(Rejected) as cancelled:
                    await engine.response(job)
                assert cancelled.value.reason == "cancelled"
                assert cancelled.value.execution == "not-started"
                assert not engine.jobs[job].get("held")
                assert journal.records[job]["reason"] == "cancelled"
                await asyncio.sleep(0.4)
                assert len(calls) == 1
        finally:
            release.set()
            await engine.close()

    asyncio.run(scenario())


def test_worker_slow_response_cannot_release_another_inference_slot(tmp_path):
    from speech_receiver.http import ReceiverMiddleware

    async def scenario():
        started_second, release_second = asyncio.Event(), asyncio.Event()
        sending_first, release_first = asyncio.Event(), asyncio.Event()
        calls = []

        async def model(scope, receive, send):
            calls.append(True)
            if len(calls) == 2:
                started_second.set()
                await release_second.wait()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"audio/wav")],
                }
            )
            await send({"type": "http.response.body", "body": b"wav"})

        middleware = ReceiverMiddleware(
            model, config(tmp_path, mode="worker"), {}, lambda _: None
        )

        async def request(slow=False):
            messages = []

            async def receive():
                return {
                    "type": "http.request",
                    "body": b'{"text":"hello"}',
                    "more_body": False,
                }

            async def send(message):
                if slow and message["type"] == "http.response.start":
                    sending_first.set()
                    await release_first.wait()
                messages.append(message)

            scope = middleware.scope("application/json", b"")
            scope["headers"].append((b"authorization", ("Bearer " + NORMAL).encode()))
            await middleware(scope, receive, send)
            return messages[0]["status"]

        first = asyncio.create_task(request(slow=True))
        second = None
        try:
            await asyncio.wait_for(sending_first.wait(), 2)
            second = asyncio.create_task(request())
            await asyncio.wait_for(started_second.wait(), 2)
            release_first.set()
            assert await first == 200
            assert await request() == 429
            assert len(calls) == 2
            release_second.set()
            assert await second == 200
            assert await request() == 200
        finally:
            release_first.set()
            release_second.set()
            await asyncio.gather(
                *[t for t in (first, second) if t is not None], return_exceptions=True
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["put", "get", "remove"])
def test_slow_blob_io_does_not_hold_state_lock_for_high_admission_or_timeout(
    tmp_path, monkeypatch, stage
):
    import threading

    async def scenario():
        entered, release = threading.Event(), threading.Event()
        running, finish_inference = asyncio.Event(), asyncio.Event()

        async def execute(body, *_):
            if body == b"hold":
                running.set()
                await finish_inference.wait()
            return 200, "audio/wav", b"wav:" + body

        engine = Engine(
            config(tmp_path, queue_timeout=0.1, retention=0.05),
            MemoryJournal(),
            execute,
        )
        await engine.start()
        slow_task = None
        result = None
        try:
            if stage == "get":
                result = await engine.submit(
                    "dev", "normal", "result", b"result", "x", b""
                )
                await engine.response(result)
                # Keep an owner during setup: retention is only 50 ms and
                # can otherwise delete the result before the delayed read starts.
                target = engine.jobs[result]["output"]
            await engine.submit("dev", "normal", "hold", b"hold", "x", b"")
            await running.wait()
            if stage == "remove":
                deleting = await engine.submit(
                    "dev", "normal", "delete", b"delete", "x", b""
                )
                target = engine.jobs[deleting]["input"]
            original = getattr(engine.blobs, stage)

            def slow(value, *args):
                if (stage == "put" and value == b"slow") or (
                    stage != "put" and value == target
                ):
                    entered.set()
                    assert release.wait(3), "unrelated state operations blocked on disk"
                return original(value, *args)

            monkeypatch.setattr(engine.blobs, stage, slow)
            if stage == "put":
                slow_task = asyncio.create_task(
                    engine.submit("dev", "normal", "slow", b"slow", "x", b"")
                )
            elif stage == "get":
                slow_task = asyncio.create_task(engine.response(result))
            else:
                slow_task = asyncio.create_task(engine.detach(deleting))
            await until(entered.is_set)
            if stage == "get":
                # The read is now pinned; remove the owner so the assertion
                # below specifically tests retention protection by that read.
                await engine.detach(result)
            async with asyncio.timeout(0.5):
                high = await engine.submit("server", "high", "high", b"high", "x", b"")
                assert engine.jobs[high]["priority"] == "high"
                expiring = await engine.submit(
                    "dev", "normal", "expiring", b"expire", "x", b""
                )
                with pytest.raises(Rejected) as timeout:
                    await engine.response(expiring)
                assert timeout.value.reason == "queue_timeout"
                assert timeout.value.execution == "not-started"
            assert not release.is_set()
            if stage == "get":
                await asyncio.sleep(0.25)
                assert result in engine.jobs  # retention cannot delete a pinned read
            release.set()
            await slow_task
            if stage == "get":
                await until(lambda: result not in engine.jobs)
        finally:
            release.set()
            finish_inference.set()
            if slow_task is not None:
                await asyncio.gather(slow_task, return_exceptions=True)
            await engine.close()

    asyncio.run(scenario())


def test_concurrent_pending_admissions_share_idempotency_and_reserve_queue_limits(
    tmp_path, monkeypatch
):
    import threading

    async def scenario():
        entered, release = threading.Event(), threading.Event()
        cfg = config(tmp_path, max_normal=1)
        engine = Engine(cfg, MemoryJournal(), lambda *_: None)
        await engine.start(dispatch=False)
        original = engine.blobs.put
        writes = []

        def slow(data):
            writes.append(data)
            if data == b"same":
                entered.set()
                assert release.wait(3)
            return original(data)

        monkeypatch.setattr(engine.blobs, "put", slow)
        first = asyncio.create_task(
            engine.submit("dev", "normal", "same", b"same", "x", b"")
        )
        duplicate = None
        try:
            await until(entered.is_set)
            duplicate = asyncio.create_task(
                engine.submit("dev", "normal", "same", b"same", "x", b"")
            )
            await asyncio.sleep(0.01)
            with pytest.raises(Rejected) as conflict:
                await engine.submit("dev", "normal", "same", b"different", "x", b"")
            assert conflict.value.status == 409
            with pytest.raises(Rejected) as full:
                await engine.submit("dev", "normal", "another", b"another", "x", b"")
            assert full.value.reason == "queue_full"
            high = await engine.submit("server", "high", "high", b"high", "x", b"")
            assert high in engine.jobs
            release.set()
            job = await first
            assert await duplicate == job
            assert engine.waiters[job] == 2
            assert writes.count(b"same") == 1
        finally:
            release.set()
            await asyncio.gather(
                *[t for t in (first, duplicate) if t is not None],
                return_exceptions=True,
            )
            await engine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["stt", "tts"])
@pytest.mark.parametrize("execution", ["finished", "not-started", None])
def test_result_quota_failure_preserves_upstream_execution_and_slot(
    tmp_path, operation, execution
):
    async def scenario():
        cfg = config(tmp_path, max_payload_bytes=5)
        cfg.operation = operation
        cfg.instances = [
            {"id": "worker", "url": "http://worker/model", "token": HIGH, "slots": 1}
        ]
        query = b"model=v3_ctc" if operation == "stt" else b""
        journal = MemoryJournal()
        engine = Engine(cfg, journal, lambda *_: None)
        calls = []

        async def worker(_):
            calls.append(True)
            if len(calls) == 1:
                # Input (2) fits, but input + result (2 + 5) exceeds quota.
                headers = (
                    {} if execution is None else {"x-voice-execution-state": execution}
                )
                return httpx.Response(
                    422 if execution == "not-started" else 200,
                    content=b"large",
                    headers=headers,
                )
            return httpx.Response(
                200, content=b"ok", headers={"x-voice-execution-state": "finished"}
            )

        await engine.http.aclose()
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(worker))
        await engine.start()
        try:
            first = await engine.submit("dev", "normal", "first", b"in", "x", query)
            expected = execution or "uncertain"
            with pytest.raises(Rejected) as failure:
                await engine.response(first)
            assert failure.value.execution == expected
            assert engine.jobs[first]["held"] == (execution is None)
            assert journal.records[first]["execution"] == expected
            assert journal.records[first]["held"] == (execution is None)
            assert engine.ready
            if execution is not None:
                await until(lambda: engine.blobs.used == 0)
                second = await engine.submit(
                    "dev", "normal", "second", b"in", "x", query
                )
                assert (await engine.response(second))[2:] == (b"ok", "finished")
                assert len(calls) == 2
        finally:
            await engine.close()
        restored = Engine(cfg, journal, lambda *_: None)
        await restored.start(dispatch=False)
        try:
            with pytest.raises(Rejected) as replay:
                await restored.response(first)
            assert replay.value.execution == expected
            assert restored.jobs[first]["held"] == (execution is None)
        finally:
            await restored.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["stt", "tts"])
@pytest.mark.parametrize("body_failure", ["oversized", "read_error"])
@pytest.mark.parametrize(
    "execution,status,expected",
    [
        ("finished", 200, "finished"),
        ("not-started", 422, "not-started"),
        (None, 200, "uncertain"),
        ("uncertain", 503, "uncertain"),
        ("finished, not-started", 200, "uncertain"),
        ("not-started", 200, "uncertain"),
    ],
)
def test_worker_body_failure_preserves_execution_and_next_slot(
    tmp_path, operation, body_failure, execution, status, expected
):
    class InterruptedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"123"
            if body_failure == "read_error":
                raise httpx.ReadError("simulated read failure")
            yield b"456"

    async def scenario():
        cfg = config(tmp_path, max_result_bytes=4)
        cfg.operation = operation
        query = b"model=v3_ctc" if operation == "stt" else b""
        cfg.instances = [
            {"id": "worker", "url": "http://worker/model", "token": HIGH, "slots": 1}
        ]
        journal = MemoryJournal()
        engine = Engine(cfg, journal, lambda *_: None)
        calls = []

        async def worker(_):
            calls.append(True)
            if len(calls) == 1:
                headers = (
                    {} if execution is None else {"x-voice-execution-state": execution}
                )
                return httpx.Response(status, headers=headers, stream=InterruptedBody())
            return httpx.Response(
                200, content=b"ok", headers={"x-voice-execution-state": "finished"}
            )

        await engine.http.aclose()
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(worker))
        await engine.start()
        try:
            first = await engine.submit("dev", "normal", "first", b"in", "x", query)
            with pytest.raises(Rejected) as failure:
                await engine.response(first)
            assert failure.value.execution == expected
            held = expected == "uncertain"
            assert engine.jobs[first]["held"] == held
            assert journal.records[first]["execution"] == expected
            assert journal.records[first]["held"] == held
            second = await engine.submit("dev", "normal", "second", b"in", "x", query)
            if held:
                await asyncio.sleep(0.025)
                assert engine.jobs[second]["state"] == "queued"
                assert len(calls) == 1
            else:
                assert (await engine.response(second))[2:] == (b"ok", "finished")
                assert len(calls) == 2
            assert engine.ready
        finally:
            await engine.close()

    asyncio.run(scenario())
