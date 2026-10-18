import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import api


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "REFERENCE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api, "_ready", False, raising=False)
    monkeypatch.setattr(api, "_initialization_error", None, raising=False)
    monkeypatch.setenv("VOXCPM_OPTIMIZE", "off")
    monkeypatch.setenv("VOXCPM_REFERENCE_CACHE_SIZE", "4")
    monkeypatch.delenv("VOXCPM_REFERENCE_WAV", raising=False)
    calls = []

    class Model:
        def __init__(self, optimize):
            self.tts_model = SimpleNamespace(sample_rate=24000, optimization_enabled=optimize)
            self.optimize = optimize

        def generate(self, **kwargs):
            calls.append((self.optimize, "wav", kwargs))
            return np.zeros(24, dtype=np.float32)

        def generate_streaming(self, **kwargs):
            calls.append((self.optimize, "stream", kwargs))
            yield np.zeros(24, dtype=np.float32)

    created = []
    def create(model_id, **kwargs):
        created.append(kwargs["optimize"])
        return Model(kwargs["optimize"])

    fake = ModuleType("voxcpm")
    fake.VoxCPM = SimpleNamespace(from_pretrained=create)
    monkeypatch.setitem(sys.modules, "voxcpm", fake)
    return calls, created, Model


def test_ready_requires_warmup_of_both_public_paths(runtime):
    calls, created, _ = runtime
    assert TestClient(api.app).get("/ready").status_code == 503
    api._load_model()
    assert TestClient(api.app).get("/ready").status_code == 200
    assert created == [False]
    assert {kind for _, kind, _ in calls} == {"wav", "stream"}
    assert all(c[2]["inference_timesteps"] == 20 for c in calls)


@pytest.mark.parametrize("mode,device,optimized", [("off", "cuda", False), ("auto", "cpu", False), ("auto", "cuda:0", True), ("on", "cuda", True)])
def test_compile_modes(runtime, monkeypatch, mode, device, optimized):
    monkeypatch.setenv("VOXCPM_OPTIMIZE", mode)
    monkeypatch.setattr(api, "DEVICE", device)
    api._load_model()
    assert runtime[1] == [optimized]


@pytest.mark.parametrize("setting,value", [("VOXCPM_OPTIMIZE", "bad"), ("VOXCPM_OPTIMIZE", "on"), ("VOXCPM_REFERENCE_CACHE_SIZE", "17"), ("VOXCPM_REFERENCE_CACHE_SIZE", "-1")])
def test_invalid_configuration_never_loads(runtime, monkeypatch, setting, value):
    monkeypatch.setattr(api, "DEVICE", "cpu")
    monkeypatch.setenv(setting, value)
    with pytest.raises((ValueError, RuntimeError)):
        api._load_model()
    assert not runtime[1]
    assert TestClient(api.app).get("/ready").status_code == 503


@pytest.mark.parametrize("mode,expected", [("auto", [True, False]), ("on", [True])])
@pytest.mark.parametrize("failure", ["swallowed", "deferred", "deferred_stream"])
def test_compile_failure_falls_back_only_in_auto(runtime, monkeypatch, mode, expected, failure):
    _, created, Model = runtime
    original_generate = Model.generate
    original_init = Model.__init__
    original_stream = Model.generate_streaming
    def init(self, optimize):
        original_init(self, optimize)
        if failure == "swallowed":
            self.tts_model.optimization_enabled = False
    def generate(self, **kwargs):
        if self.optimize and failure == "deferred":
            raise RuntimeError("deferred compile failure")
        return original_generate(self, **kwargs)
    def stream(self, **kwargs):
        if self.optimize and failure == "deferred_stream":
            raise RuntimeError("deferred streaming compile failure")
        yield from original_stream(self, **kwargs)
    monkeypatch.setattr(Model, "__init__", init)
    monkeypatch.setattr(Model, "generate", generate)
    monkeypatch.setattr(Model, "generate_streaming", stream)
    monkeypatch.setattr(api, "DEVICE", "cuda")
    monkeypatch.setenv("VOXCPM_OPTIMIZE", mode)
    if mode == "on":
        with pytest.raises(RuntimeError):
            api._load_model()
        with pytest.raises(RuntimeError):
            api._load_model()
    else:
        api._load_model()
        api._load_model()
    assert created == expected
    assert TestClient(api.app).get("/ready").status_code == (200 if mode == "auto" else 503)


def test_pinned_voice_ignores_target_parameters(runtime, monkeypatch, tmp_path):
    path = tmp_path / "pinned.wav"
    sf.write(path, np.zeros(24), 24000)
    monkeypatch.setenv("VOXCPM_REFERENCE_WAV", str(path))
    model = runtime[2](False)
    for steps in (10, 20):
        assert api._reference_for(model, api.SynthesisRequest(text="Ответ", seed=42, cfg_value=3, inference_timesteps=steps)) == str(path)
    assert api._reference_for(model, api.SynthesisRequest(text="Ответ", use_reference=False)) is None
    assert not runtime[0]


@pytest.mark.parametrize("content", [None, b"broken"])
def test_bad_pinned_voice_blocks_readiness_without_replacement(runtime, monkeypatch, tmp_path, content):
    path = tmp_path / "pinned.wav"
    if content is not None:
        path.write_bytes(content)
    monkeypatch.setenv("VOXCPM_REFERENCE_WAV", str(path))
    with pytest.raises((ValueError, RuntimeError, OSError)):
        api._load_model()
    assert not runtime[0]
    assert TestClient(api.app).get("/ready").status_code == 503


@pytest.mark.parametrize("format,rate,shape", [("FLAC", 24000, (24,)), ("WAV", 16000, (24,)), ("WAV", 24000, (24, 2))])
def test_pinned_voice_rejects_wrong_format_rate_or_channels(runtime, monkeypatch, tmp_path, format, rate, shape):
    path = tmp_path / "pin.wav"
    sf.write(path, np.zeros(shape), rate, format=format)
    monkeypatch.setenv("VOXCPM_REFERENCE_WAV", str(path))
    with pytest.raises((ValueError, RuntimeError)):
        api._load_model()
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
    path = tmp_path / "voice.wav"
    sf.write(path, np.zeros(24), 24000)
    monkeypatch.setenv("VOXCPM_REFERENCE_WAV", str(path))
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
            kwargs["prompt_cache"]["feature"][0] = 99
            return np.zeros(24)
    model = Model()
    monkeypatch.setattr(api, "_load_model", lambda: model)
    client = TestClient(api.app)
    assert client.post("/synthesize", json={"text": "а" * 1001}).status_code == 200
    assert client.post("/synthesize", json={"text": "Ответ", "inference_timesteps": 10}).status_code == 200
    assert len(builds) == 1
    assert len(calls) == 4
    assert all("retry_badcase" not in call for call in calls)
