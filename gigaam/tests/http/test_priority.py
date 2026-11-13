"""Receiver wraps the production ASR handler, not an alternative model adapter."""

from api import create_app, resolve_model_name
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
    app = install(create_app(model_loader=FakeModel), cfg, resolve_model_name, Memory)
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


def test_default_and_alias_are_canonical_before_worker_selection(tmp_path, monkeypatch):
    import api
    import httpx

    monkeypatch.setattr(api, "MODEL_NAME", "e2e_rnnt")
    token = "local-development-client-token-0001"
    cfg = Config(
        operation="stt",
        data_dir=str(tmp_path),
        brokers=["unused"],
        topic="test",
        clients={"dev": {"token": token, "priority": "normal"}},
        instances=[
            {
                "id": "ctc",
                "url": "http://ctc/transcribe",
                "token": token,
                "slots": 1,
                "models": ["v3_ctc"],
            },
            {
                "id": "rnnt",
                "url": "http://rnnt/transcribe",
                "token": token,
                "slots": 1,
                "models": ["v3_e2e_rnnt"],
            },
        ],
    )
    app = install(create_app(preload=False), cfg, resolve_model_name, Memory)
    calls = []

    async def worker(request):
        calls.append((request.url.host, request.url.params["model"]))
        return httpx.Response(
            200,
            json={"text": "ответ", "model": request.url.params["model"]},
            headers={"x-voice-execution-state": "finished"},
        )

    with TestClient(app) as client:
        client.portal.call(app.state.receiver.http.aclose)
        app.state.receiver.http = httpx.AsyncClient(
            transport=httpx.MockTransport(worker)
        )
        headers = {"Authorization": "Bearer " + token}
        for key, query, selected in (
            ("default", "", "v3_e2e_rnnt"),
            ("alias", "?model=ctc", "v3_ctc"),
            ("alias", "?model=v3_ctc", "v3_ctc"),
            ("default", "?model=v3_e2e_rnnt", "v3_e2e_rnnt"),
        ):
            response = client.post(
                "/transcribe" + query,
                files={"file": ("audio.wav", b"audio")},
                headers=dict(headers, **{"Idempotency-Key": key}),
            )
            assert response.status_code == 200 and response.json()["model"] == selected
        assert calls == [("rnnt", "v3_e2e_rnnt"), ("ctc", "v3_ctc")]
        assert {j["model"] for j in app.state.receiver.jobs.values()} == {
            "v3_ctc",
            "v3_e2e_rnnt",
        }


def test_default_model_without_compatible_worker_rejected_before_persistence(
    tmp_path, monkeypatch
):
    import api

    monkeypatch.setattr(api, "MODEL_NAME", "v3_e2e_rnnt")
    token = "local-development-client-token-0001"
    cfg = Config(
        operation="stt",
        data_dir=str(tmp_path),
        brokers=["unused"],
        topic="test",
        clients={"dev": {"token": token, "priority": "normal"}},
        instances=[
            {
                "id": "ctc",
                "url": "http://ctc/transcribe",
                "token": token,
                "slots": 1,
                "models": ["v3_ctc"],
            }
        ],
    )
    app = install(create_app(preload=False), cfg, resolve_model_name, Memory)
    with TestClient(app) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("audio.wav", b"audio")},
            headers={"Authorization": "Bearer " + token},
        )
        assert response.status_code == 422
        assert response.json() == {"error": "no_compatible_model"}
        assert app.state.receiver.jobs == {}
        assert list(tmp_path.glob("*.blob")) == []
