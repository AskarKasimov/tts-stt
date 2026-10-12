import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import api


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "REFERENCE_CACHE_DIR", tmp_path, raising=False)
    return tmp_path


class RecordingModel:
    tts_model = type("TTS", (), {"sample_rate": 48000})()

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        reference = kwargs.get("reference_wav_path")
        if reference:
            audio, rate = sf.read(reference)
            assert rate == 48000
            assert len(audio) == 480
        return np.full(480, 0.1 if reference else 0.2, dtype=np.float32)


def test_default_request_creates_reference_then_returns_only_target_audio(monkeypatch, cache_dir):
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    response = TestClient(api.app).post("/synthesize", json={"text": "Нужный ответ."})
    assert response.status_code == 200
    assert len(model.calls) == 2
    reference, target = model.calls
    assert reference["text"].startswith("Давайте внимательно разберём задачу.")
    assert reference["text"].endswith("спокойно продолжайте работу.")
    assert reference["seed"] == target["seed"] == 510
    assert reference["cfg_value"] == target["cfg_value"] == 2
    assert reference["inference_timesteps"] == target["inference_timesteps"] == 20
    assert reference["retry_badcase"] is False
    assert "reference_wav_path" not in reference
    assert target["text"] == "Нужный ответ."
    assert Path(target["reference_wav_path"]).parent == cache_dir
    audio, _ = sf.read(io.BytesIO(response.content))
    assert len(audio) == 480
    assert np.allclose(audio, 0.1, atol=1e-4)


def test_reference_is_reused_after_model_restart_and_float_normalization(monkeypatch, cache_dir):
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    client = TestClient(api.app)
    assert client.post("/synthesize", json={"text": "Первый", "cfg_value": 2}).status_code == 200
    original = model.calls[-1].get("reference_wav_path")
    restarted = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: restarted)
    assert client.post("/synthesize", json={"text": "Другой", "cfg_value": 2.0}).status_code == 200
    assert len(restarted.calls) == 1
    assert restarted.calls[0]["reference_wav_path"] == original
    assert len(list(cache_dir.glob("*.wav"))) == 1


@pytest.mark.parametrize("change", [
    {"seed": 511}, {"cfg_value": 2.5}, {"inference_timesteps": 21},
])
def test_each_parameter_selects_a_different_reference(monkeypatch, cache_dir, change):
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    client = TestClient(api.app)
    assert client.post("/synthesize", json={"text": "Первый"}).status_code == 200
    assert client.post("/synthesize", json={"text": "Второй", **change}).status_code == 200
    assert len(list(cache_dir.glob("*.wav"))) == 2
    assert model.calls[1]["reference_wav_path"] != model.calls[3]["reference_wav_path"]


@pytest.mark.parametrize("constant", ["MODEL_ID", "REFERENCE_TEXT"])
def test_model_or_reference_text_change_invalidates_cache(monkeypatch, cache_dir, constant):
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    client = TestClient(api.app)
    assert client.post("/synthesize", json={"text": "Первый"}).status_code == 200
    monkeypatch.setattr(api, constant, "changed")
    assert client.post("/synthesize", json={"text": "Второй"}).status_code == 200
    assert len(list(cache_dir.glob("*.wav"))) == 2


def test_disabled_reference_does_not_read_or_create_cache(monkeypatch, cache_dir):
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    cache_dir.rmdir()
    response = TestClient(api.app).post("/synthesize", json={
        "text": "Ответ", "use_reference": False, "seed": None,
    })
    assert response.status_code == 200
    assert len(model.calls) == 1
    assert "reference_wav_path" not in model.calls[0]
    assert model.calls[0]["seed"] is None
    assert not cache_dir.exists()


def test_long_text_uses_one_reference_for_every_chunk(monkeypatch, cache_dir):
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    response = TestClient(api.app).post("/synthesize", json={"text": "а" * 1001})
    assert response.status_code == 200
    assert len(model.calls) == 4
    paths = [call["reference_wav_path"] for call in model.calls[1:]]
    assert len(set(paths)) == 1
    assert [len(call["text"]) for call in model.calls[1:]] == [500, 500, 1]


def test_failed_reference_is_not_cached_and_next_request_can_retry(monkeypatch, cache_dir):
    class FailingModel(RecordingModel):
        def generate(self, **kwargs):
            raise RuntimeError("Reference synthesis failed")

    monkeypatch.setattr(api, "_load_model", FailingModel)
    client = TestClient(api.app, raise_server_exceptions=False)
    assert client.post("/synthesize", json={"text": "Ответ"}).status_code == 500
    assert not list(cache_dir.iterdir())
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    assert client.post("/synthesize", json={"text": "Ответ"}).status_code == 200
    assert len(model.calls) == 2


def test_partial_reference_write_is_cleaned_up(monkeypatch, cache_dir):
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)

    def failed_write(path, *args, **kwargs):
        Path(path).write_bytes(b"partial WAV")
        raise OSError("disk full")

    monkeypatch.setattr(api.sf, "write", failed_write)
    response = TestClient(api.app, raise_server_exceptions=False).post(
        "/synthesize", json={"text": "Ответ"},
    )
    assert response.status_code == 500
    assert not list(cache_dir.iterdir())


def test_corrupt_cached_reference_is_regenerated(monkeypatch, cache_dir):
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    client = TestClient(api.app)
    assert client.post("/synthesize", json={"text": "Ответ"}).status_code == 200
    Path(model.calls[-1]["reference_wav_path"]).write_bytes(b"broken")
    model.calls.clear()
    assert client.post("/synthesize", json={"text": "Ответ"}).status_code == 200
    assert len(model.calls) == 2


def test_concurrent_requests_create_one_reference(monkeypatch, cache_dir):
    model = RecordingModel()
    monkeypatch.setattr(api, "_load_model", lambda: model)

    def request():
        with TestClient(api.app) as client:
            return client.post("/synthesize", json={"text": "Ответ"}).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: request(), range(2))) == [200, 200]
    assert len(model.calls) == 3
    assert len(list(cache_dir.glob("*.wav"))) == 1


def test_reference_with_null_seed_is_rejected_before_loading_model(monkeypatch, cache_dir):
    def unexpected_load():
        raise AssertionError("Invalid request loaded the model")

    monkeypatch.setattr(api, "_load_model", unexpected_load)
    assert TestClient(api.app).post("/synthesize", json={
        "text": "Ответ", "seed": None, "use_reference": True,
    }).status_code == 422
