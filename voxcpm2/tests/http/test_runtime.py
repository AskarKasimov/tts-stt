import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "REFERENCE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api, "_ready", False, raising=False)
    monkeypatch.setattr(api, "_initialization_error", None, raising=False)
    monkeypatch.setenv("VOXCPM_REFERENCE_CACHE_SIZE", "4")
    calls = []

    class Model:
        def __init__(self):
            self.tts_model = SimpleNamespace(sample_rate=24000, optimization_enabled=False)
            self.optimize = False

        def generate(self, **kwargs):
            calls.append((self.optimize, "wav", kwargs))
            return np.zeros(24, dtype=np.float32)

        def generate_streaming(self, **kwargs):
            calls.append((self.optimize, "stream", kwargs))
            yield np.zeros(24, dtype=np.float32)

    created = []
    def create(model_id, **kwargs):
        created.append(kwargs["optimize"])
        return Model()

    fake = ModuleType("voxcpm")
    fake.VoxCPM = SimpleNamespace(from_pretrained=create)
    monkeypatch.setitem(sys.modules, "voxcpm", fake)
    return calls, created, Model


def test_ready_requires_model_load_but_does_not_run_startup_inference(runtime):
    calls, created, _ = runtime
    assert TestClient(api.app).get("/ready").status_code == 503
    api._load_model()
    assert TestClient(api.app).get("/ready").status_code == 200
    assert created == [False]
    assert calls == []


def test_model_load_uses_eager_mode_on_any_device(runtime, monkeypatch):
    monkeypatch.setattr(api, "DEVICE", "cuda:0")
    api._load_model()
    assert runtime[1] == [False]


@pytest.mark.parametrize("setting,value", [("VOXCPM_REFERENCE_CACHE_SIZE", "17"), ("VOXCPM_REFERENCE_CACHE_SIZE", "-1")])
def test_invalid_configuration_never_loads(runtime, monkeypatch, setting, value):
    monkeypatch.setattr(api, "DEVICE", "cpu")
    monkeypatch.setenv(setting, value)
    with pytest.raises((ValueError, RuntimeError)):
        api._load_model()
    assert not runtime[1]
    assert TestClient(api.app).get("/ready").status_code == 503


def test_reference_feature_cache_is_content_keyed_bounded_and_isolated(tmp_path):
    from runtime import ReferenceFeatures
    path = tmp_path / "voice.wav"
    path.write_bytes(b"one")
    builds = []
    class TTS:
        device = "cpu"
        def build_prompt_cache(self, **kwargs):
            builds.append(path.read_bytes())
            return {"feature": np.array([len(builds)])}
    model = SimpleNamespace(tts_model=TTS())
    cache = ReferenceFeatures(1)
    first, hit = cache.get(model, str(path), "revision")
    assert not hit
    first["feature"][0] = 100
    second, hit = cache.get(model, str(path), "revision")
    assert hit and second["feature"][0] == 1
    path.write_bytes(b"two")
    assert not cache.get(model, str(path), "revision")[1]
    path.write_bytes(b"one")
    assert not cache.get(model, str(path), "revision")[1]
    assert len(cache) == 1
    other = SimpleNamespace(tts_model=TTS())
    assert not cache.get(other, str(path), "revision")[1]
    assert len(cache) == 1
    disabled = ReferenceFeatures(0)
    assert not disabled.get(model, str(path), "revision")[1]
    assert not disabled.get(model, str(path), "revision")[1]
    assert len(disabled) == 0


def test_invalid_features_are_not_retained(tmp_path):
    from runtime import ReferenceFeatures
    path = tmp_path / "voice.wav"
    path.write_bytes(b"voice")
    tts = SimpleNamespace(build_prompt_cache=lambda **kwargs: {"feature": np.array([np.nan])})
    cache = ReferenceFeatures(4)
    with pytest.raises(RuntimeError, match="feature"):
        cache.get(SimpleNamespace(tts_model=tts), str(path), "revision")
    assert len(cache) == 0


def test_reference_rewritten_during_encoding_is_not_cached(tmp_path):
    from runtime import ReferenceFeatures
    path = tmp_path / "voice.wav"
    path.write_bytes(b"one")
    def build(**kwargs):
        path.write_bytes(b"two")
        return {"feature": np.zeros(1)}
    cache = ReferenceFeatures(4)
    with pytest.raises(RuntimeError, match="changed"):
        cache.get(SimpleNamespace(tts_model=SimpleNamespace(build_prompt_cache=build)), str(path), "revision")
    assert len(cache) == 0


@pytest.mark.parametrize("audio", [np.array([np.nan]), np.array([])])
def test_wav_rejects_invalid_waveform(monkeypatch, audio):
    model = SimpleNamespace(tts_model=SimpleNamespace(sample_rate=24000), generate=lambda **kwargs: audio)
    monkeypatch.setattr(api, "_load_model", lambda: model)
    for text in ("Ответ", "а" * 501):
        response = TestClient(api.app, raise_server_exceptions=False).post("/synthesize", json={"text": text, "use_reference": False})
        assert response.status_code == 500


def test_longform_reuses_prepared_features_and_wav_keeps_retry_default(monkeypatch, tmp_path):
    from runtime import ReferenceFeatures
    monkeypatch.setattr(api, "REFERENCE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(api, "_features", ReferenceFeatures(4))
    builds, calls = [], []
    class TTS:
        sample_rate = 24000
        def build_prompt_cache(self, **kwargs):
            builds.append(kwargs)
            return {"feature": np.array([1])}
    class Model:
        tts_model = TTS()
        def generate(self, **kwargs):
            calls.append(kwargs)
            if "prompt_cache" in kwargs:
                kwargs["prompt_cache"]["feature"][0] = 99
            return np.zeros(24)
    model = Model()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    client = TestClient(api.app)
    assert client.post("/synthesize", json={"text": "а" * 1001}).status_code == 200
    assert client.post("/synthesize", json={"text": "Ответ"}).status_code == 200
    assert len(builds) == 1
    assert len(calls) == 5
    target_calls = [call for call in calls if "prompt_cache" in call]
    assert len(target_calls) == 4
    assert all("retry_badcase" not in call for call in target_calls)
