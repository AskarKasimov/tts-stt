import asyncio
import base64
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api


class Model:
    tts_model = SimpleNamespace(sample_rate=24000)
    def generate(self, **kwargs):
        return np.zeros(24)
    def generate_streaming(self, **kwargs):
        self.options = kwargs
        yield np.array([-2, 0, 2], dtype=np.float32)
        yield np.zeros(40000, dtype=np.float32)


@pytest.fixture
def model(monkeypatch):
    model = Model()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    monkeypatch.setattr(api, "_initialization_error", None)
    return model


def events(client, **kwargs):
    response = client.post("/synthesize/stream", json={"text": "Ответ", "use_reference": False, **kwargs})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.headers["cache-control"] == "no-store"
    assert "content-length" not in response.headers
    assert all(len(line) < 256 * 1024 for line in response.content.splitlines())
    return [json.loads(line) for line in response.iter_lines()]


def test_stream_pcm_sequence_exact_samples_and_clipping(model):
    result = events(TestClient(api.app))
    assert result[0] == {"type": "start", "version": 1, "sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"}
    audio = result[1:-1]
    assert [item["seq"] for item in audio] == list(range(len(audio)))
    blocks = [base64.b64decode(item["data"], validate=True) for item in audio]
    assert all(0 < len(block) <= 65536 and len(block) % 2 == 0 for block in blocks)
    pcm = np.frombuffer(b"".join(blocks), dtype="<i2")
    assert pcm[:3].tolist() == [-32768, 0, 32767]
    assert result[-1] == {"type": "end", "chunks": len(audio), "samples": len(pcm)}
    assert len(pcm) == 40003
    assert model.options["retry_badcase"] is False


def test_longform_stream_includes_exact_pause_and_no_retries(model):
    result = events(TestClient(api.app), text="а" * 501)
    pcm = np.frombuffer(b"".join(base64.b64decode(e["data"]) for e in result if e["type"] == "audio"), dtype="<i2")
    assert len(pcm) == 40003 * 2 + 3600
    assert np.all(pcm[40003:43603] == 0)
    assert result[-1]["samples"] == len(pcm)


@pytest.mark.parametrize("failure", ["before", "after", "nan", "empty"])
def test_stream_errors_are_generic_terminal_without_end(model, monkeypatch, failure):
    def generate(**kwargs):
        if failure == "after":
            yield np.zeros(4)
        if failure == "nan":
            yield np.array([np.nan])
        if failure != "empty":
            raise RuntimeError("secret failure details")
    monkeypatch.setattr(model, "generate_streaming", generate)
    result = events(TestClient(api.app))
    assert result[-1] == {"type": "error", "code": "SYNTHESIS_FAILED"}
    assert all(e["type"] != "end" for e in result)
    assert "secret" not in json.dumps(result)


def test_reference_failure_after_open_still_has_one_start(model, monkeypatch):
    monkeypatch.setattr(api, "_reference_for", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("reference failed")))
    result = events(TestClient(api.app), use_reference=True)
    assert [item["type"] for item in result] == ["start", "error"]


def test_stream_validation_and_configuration_errors_remain_http(model, monkeypatch):
    client = TestClient(api.app)
    assert client.post("/synthesize/stream", json={"text": ""}).status_code == 422
    monkeypatch.setenv("VOXCPM_REFERENCE_CACHE_SIZE", "invalid")
    assert client.post("/synthesize/stream", json={"text": "Ответ"}).status_code == 503


def test_loading_runtime_does_not_accept_user_traffic(model, monkeypatch):
    monkeypatch.setattr(api, "_startup_started", True, raising=False)
    monkeypatch.setattr(api, "_ready", False)
    client = TestClient(api.app)
    for route in ("/synthesize", "/synthesize/stream"):
        assert client.post(route, json={"text": "Ответ", "use_reference": False}).status_code == 503


def test_stream_decoded_limit_produces_error_not_success(model, monkeypatch):
    import streaming
    monkeypatch.setattr(streaming, "MAX_PCM_BYTES", 100)
    result = events(TestClient(api.app))
    assert result[-1]["type"] == "error"


def test_stream_metrics_count_completed_output_once_and_reject_partial(model, monkeypatch):
    from test_metrics import current, sample_value
    client = TestClient(api.app)
    before = current(client)
    events(client)
    after = current(client)
    assert sample_value(after, "speech_requests_total", outcome="success") - sample_value(before, "speech_requests_total", outcome="success") == 1
    assert sample_value(after, "speech_audio_seconds_total", direction="output") - sample_value(before, "speech_audio_seconds_total", direction="output") == pytest.approx(40003 / 24000)
    def partial(**kwargs):
        yield np.zeros(4)
        raise RuntimeError("failure")
    monkeypatch.setattr(model, "generate_streaming", partial)
    events(client)
    failed = current(client)
    assert sample_value(failed, "speech_requests_total", outcome="server_error") - sample_value(after, "speech_requests_total", outcome="server_error") == 1
    assert sample_value(failed, "speech_audio_seconds_total", direction="output") == sample_value(after, "speech_audio_seconds_total", direction="output")


def test_bounded_slow_reader_releases_model_and_closes_generator(model, monkeypatch):
    from streaming import StreamJob
    closed = threading.Event()
    def generate(**kwargs):
        try:
            while True:
                yield np.zeros(32768)
        finally:
            closed.set()
    monkeypatch.setattr(model, "generate_streaming", generate)
    monkeypatch.setattr(api, "STREAM_DRAIN_TIMEOUT", 0.05)
    job = api._stream_job(api.SynthesisRequest(text="Ответ", use_reference=False))
    assert isinstance(job, StreamJob)
    assert job.done.wait(2)
    assert closed.is_set()
    assert job.queue.qsize() <= 8
    assert all(len(item) <= 90000 for item in list(job.queue.queue))
    assert api._model_lock.acquire(timeout=1)
    api._model_lock.release()
    assert TestClient(api.app).post("/synthesize", json={"text": "Следующий", "use_reference": False}).status_code == 200


def test_cancellation_while_waiting_for_model_does_not_generate(model):
    api._model_lock.acquire()
    try:
        job = api._stream_job(api.SynthesisRequest(text="Ответ", use_reference=False))
        job.cancel.set()
        assert job.done.wait(1)
        assert not hasattr(model, "options")
    finally:
        api._model_lock.release()


def test_actual_asgi_stream_flushes_before_generation_finishes_and_disconnect_closes(model, monkeypatch):
    first_sent = threading.Event()
    closed = threading.Event()
    def generate(**kwargs):
        try:
            yield np.zeros(24)
            assert first_sent.wait(2), "First PCM must flush before synthesis completes"
            while not kwargs["cancel_event"].is_set():
                time.sleep(0.01)
                yield np.zeros(24)
        finally:
            closed.set()
    monkeypatch.setattr(model, "generate_streaming", generate)
    async def run():
        received = False
        disconnect = asyncio.Event()
        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": json.dumps({"text": "Ответ", "use_reference": False}).encode(), "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}
        async def send(message):
            if message["type"] == "http.response.body" and b'"audio"' in message.get("body", b""):
                first_sent.set()
                disconnect.set()
        await api.app({"type": "http", "method": "POST", "path": "/synthesize/stream", "raw_path": b"/synthesize/stream", "query_string": b"", "scheme": "http", "http_version": "1.1", "headers": [(b"content-type", b"application/json")], "server": ("test", 80), "client": ("test", 123)}, receive, send)
    asyncio.run(run())
    assert first_sent.is_set()
    assert closed.wait(2)
    assert api._model_lock.acquire(timeout=1)
    api._model_lock.release()


def test_blocked_transport_is_cancelled_within_drain_timeout(model, monkeypatch):
    from test_metrics import current, sample_value
    client = TestClient(api.app)
    before = current(client)
    monkeypatch.setattr(api, "STREAM_DRAIN_TIMEOUT", 0.05)
    closed = threading.Event()
    def generate(**kwargs):
        try:
            while True:
                yield np.zeros(32768)
        finally:
            closed.set()
    monkeypatch.setattr(model, "generate_streaming", generate)
    async def run():
        received = False
        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": b'{"text":"hello","use_reference":false}', "more_body": False}
            await asyncio.Event().wait()
        async def send(message):
            if message["type"] == "http.response.body":
                await asyncio.Event().wait()
        scope = {"type": "http", "method": "POST", "path": "/synthesize/stream", "raw_path": b"/synthesize/stream", "query_string": b"", "scheme": "http", "http_version": "1.1", "headers": [(b"content-type", b"application/json")], "server": ("test", 80), "client": ("test", 123)}
        try:
            await asyncio.wait_for(api.app(scope, receive, send), timeout=1)
        except TimeoutError:
            assert time.monotonic() - started < 0.5
    started = time.monotonic()
    asyncio.run(run())
    assert closed.wait(1)
    assert api._model_lock.acquire(timeout=1)
    api._model_lock.release()
    after = current(client)
    assert sample_value(after, "speech_requests_total", outcome="server_error") - sample_value(before, "speech_requests_total", outcome="server_error") == 1
