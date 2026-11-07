import api
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


def test_tts_through_priority_receiver(tmp_path, monkeypatch):
    token = "local-development-client-token-0001"
    monkeypatch.setattr(api, "_load_model", lambda: FakeModel())
    # Fresh app/module to avoid adding middleware to another test's running app.
    import importlib.util

    spec = importlib.util.spec_from_file_location("queued_tts_api", api.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_load_model", lambda: FakeModel())
    cfg = Config(
        operation="tts",
        data_dir=str(tmp_path),
        brokers=["unused"],
        topic="test",
        clients={"dev": {"token": token, "priority": "normal"}},
        instances=[],
    )
    app = install(module.app, cfg, module.SynthesisRequest.model_validate, Memory)
    with TestClient(app) as client:
        assert client.post("/synthesize", json={"text": "Привет"}).status_code == 401
        response = client.post(
            "/synthesize",
            json={"text": "Привет", "use_reference": False},
            headers={"Authorization": "Bearer " + token},
        )
        assert response.status_code == 200 and response.content.startswith(b"RIFF")
        assert response.headers["x-voice-execution-state"] == "finished"
