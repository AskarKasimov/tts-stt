import numpy as np
import io
import soundfile as sf
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


def test_longform_synthesizes_all_chunks_into_one_wav(monkeypatch):
    class RecordingModel(FakeModel):
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return np.full(480, len(self.calls) / 10, dtype=np.float32)

    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    text = "А" * 490 + ". " + "Б" * 20
    response = TestClient(api.app).post(
        "/synthesize/longform",
        json={"text": text, "seed": 42, "cfg_value": 3.0, "inference_timesteps": 12},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    audio, sample_rate = sf.read(io.BytesIO(response.content))
    assert sample_rate == 48000
    assert len(audio) == 480 * 2 + 7200
    assert np.allclose(audio[:480], 0.1, atol=1e-4)
    assert np.all(audio[480:7680] == 0)
    assert np.allclose(audio[7680:], 0.2, atol=1e-4)
    assert [call["text"] for call in model.calls] == ["А" * 490 + ".", "Б" * 20]
    assert [call["seed"] for call in model.calls] == [42, 42]
    assert all(call["cfg_value"] == 3.0 for call in model.calls)
    assert all(call["inference_timesteps"] == 12 for call in model.calls)


def test_longform_splits_on_spaces_without_losing_words(monkeypatch):
    chunks = []

    class RecordingModel(FakeModel):
        def generate(self, **kwargs):
            chunks.append(kwargs["text"])
            return np.zeros(480, dtype=np.float32)

    monkeypatch.setattr(api, "_load_model", RecordingModel)
    text = ("слово " * 120).strip()
    response = TestClient(api.app).post("/synthesize/longform", json={"text": text})
    assert response.status_code == 200
    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert " ".join(chunks) == text


def test_longform_splits_a_word_longer_than_500_characters(monkeypatch):
    chunks = []

    class RecordingModel(FakeModel):
        def generate(self, **kwargs):
            chunks.append(kwargs["text"])
            return np.zeros(480, dtype=np.float32)

    monkeypatch.setattr(api, "_load_model", RecordingModel)
    short = TestClient(api.app).post("/synthesize", json={"text": "а" * 501})
    response = TestClient(api.app).post(
        "/synthesize/longform", json={"text": "а" * 1001}
    )
    assert short.status_code == 422
    assert response.status_code == 200
    assert [len(chunk) for chunk in chunks] == [500, 500, 1]
    assert "".join(chunks) == "а" * 1001


def test_longform_accepts_exactly_10000_characters(monkeypatch):
    chunks = []

    class RecordingModel(FakeModel):
        def generate(self, **kwargs):
            chunks.append(kwargs["text"])
            return np.zeros(480, dtype=np.float32)

    monkeypatch.setattr(api, "_load_model", RecordingModel)
    response = TestClient(api.app).post(
        "/synthesize/longform", json={"text": "а" * 10000}
    )
    assert response.status_code == 200
    assert len(chunks) == 20
    assert all(len(chunk) == 500 for chunk in chunks)


def test_longform_rejects_text_over_10000_characters_and_blank_text(monkeypatch):
    def unexpected_model_load():
        raise AssertionError("Invalid text must not load the model")

    monkeypatch.setattr(api, "_load_model", unexpected_model_load)
    client = TestClient(api.app)
    assert client.post("/synthesize/longform", json={"text": "а" * 10001}).status_code == 422
    assert client.post("/synthesize/longform", json={"text": "  "}).status_code == 422


def test_longform_fails_without_returning_partial_audio(monkeypatch):
    class FailingModel(FakeModel):
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("Generation failed")
            return np.zeros(480, dtype=np.float32)

    monkeypatch.setattr(api, "_load_model", FailingModel)
    client = TestClient(api.app, raise_server_exceptions=False)
    response = client.post("/synthesize/longform", json={"text": "а" * 501})
    assert response.status_code == 500
    assert response.headers["content-type"] != "audio/wav"
