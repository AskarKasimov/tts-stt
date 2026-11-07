"""Receiver wraps the production ASR handler, not an alternative model adapter."""

from api import create_app
from fastapi.testclient import TestClient
from speech_receiver.config import Config
from speech_receiver.http import install
from test_api import FakeModel


class Memory:
    def __init__(self, *_):
        self.records = {}

    def replay(self):
        return dict(self.records)

    def append(self, record, delete=False):
        if delete:
            self.records.pop(record["id"], None)
        else:
            self.records[record["id"]] = dict(record)

    def close(self):
        pass


def test_asr_through_priority_receiver(tmp_path):
    token = "local-development-client-token-0001"
    cfg = Config(
        operation="stt",
        data_dir=str(tmp_path),
        brokers=["unused"],
        topic="test",
        clients={"dev": {"token": token, "priority": "normal"}},
        instances=[],
    )
    app = install(create_app(model_loader=FakeModel), cfg, lambda _: None, Memory)
    with TestClient(app) as client:
        assert (
            client.post(
                "/transcribe", files={"file": ("audio.wav", b"audio")}
            ).status_code
            == 401
        )
        headers = {"Authorization": "Bearer " + token, "Idempotency-Key": "same"}
        first = client.post(
            "/transcribe", files={"file": ("audio.wav", b"audio")}, headers=headers
        )
        assert first.status_code == 200
        # A new multipart boundary must not break idempotency.
        second = client.post(
            "/transcribe", files={"file": ("audio.wav", b"audio")}, headers=headers
        )
        assert second.status_code == 200 and second.json() == first.json()
        metrics = client.get("/metrics")
        assert (
            "receiver_jobs_total" in metrics.text
            and "speech_requests_total" in metrics.text
        )
