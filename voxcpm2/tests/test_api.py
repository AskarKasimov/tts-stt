import numpy as np
import sys
from types import ModuleType
from fastapi.testclient import TestClient

import api


class FakeModel:
    tts_model = type("TTS", (), {"sample_rate": 48000})()

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return np.zeros(480, dtype=np.float32)


def test_synthesize_returns_wav(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    response = TestClient(api.app).post("/synthesize", json={"text": "Привет, мир!"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert model.kwargs["text"] == "Привет, мир!"


def test_synthesize_rejects_empty_text():
    response = TestClient(api.app).post("/synthesize", json={"text": ""})
    assert response.status_code == 422
    response = TestClient(api.app).post("/synthesize", json={"text": "   "})
    assert response.status_code == 422


def test_model_loader_uses_configured_device(monkeypatch):
    fake_voxcpm = ModuleType("voxcpm")
    calls = {}

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.update(model_id=model_id, **kwargs)
            return FakeModel()

    fake_voxcpm.VoxCPM = FakeVoxCPM
    monkeypatch.setitem(sys.modules, "voxcpm", fake_voxcpm)
    monkeypatch.setattr(api, "DEVICE", "cuda")
    monkeypatch.setattr(api, "_model", None)
    api._load_model()
    assert calls["model_id"] == api.MODEL_ID
    assert calls["device"] == "cuda"
