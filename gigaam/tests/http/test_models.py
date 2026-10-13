"""Model selection and cache behavior without downloading checkpoints."""

import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Barrier

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import api


@dataclass
class Result:
    text: str
    words: list | None = None


@dataclass
class LongResult(Result):
    segments: tuple = ()


class Model:
    def __init__(self, name, generation):
        self.text = f"{name}:{generation}"

    def transcribe(self, path, word_timestamps=False):
        if Path(path).read_bytes() == b"long":
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")
        return Result(self.text, [] if word_timestamps else None)

    def transcribe_longform(self, path, word_timestamps=False):
        return LongResult(self.text)


class Loader:
    def __init__(self):
        self.loads = []
        self.references = []

    def __call__(self, name=api.MODEL_NAME):
        self.loads.append(name)
        model = Model(name, self.loads.count(name))
        self.references.append(weakref.ref(model))
        return model


def transcribe(client, model=None):
    params = {} if model is None else {"model": model}
    return client.post(
        "/transcribe", params=params, files={"file": ("voice.wav", b"audio")}
    )


def test_selected_model_is_reused_and_default_stays_compatible():
    loader = Loader()
    with TestClient(api.create_app(model_loader=loader)) as client:
        assert transcribe(client).json()["text"] == f"{api.MODEL_NAME}:1"
        for _ in range(2):
            assert transcribe(client, "multilingual_large_ctc").json() == {
                "text": "multilingual_large_ctc:1",
                "model": "multilingual_large_ctc",
            }
        assert transcribe(client).json()["text"] == f"{api.MODEL_NAME}:1"
    assert loader.loads == [api.MODEL_NAME, "multilingual_large_ctc"]


@pytest.mark.parametrize("name", ["unknown", "emo", "v3_ssl", "/tmp/model.ckpt", ""])
def test_invalid_asr_model_is_rejected_before_loading(name):
    loader = Loader()
    with TestClient(api.create_app(model_loader=loader)) as client:
        assert transcribe(client, name).status_code == 422
    assert loader.loads == [api.MODEL_NAME]


def test_alias_shares_cache_with_canonical_name():
    loader = Loader()
    with TestClient(api.create_app(model_loader=loader)) as client:
        assert transcribe(client, "e2e_rnnt").json()["model"] == "v3_e2e_rnnt"
        assert transcribe(client, "v3_e2e_rnnt").json()["text"] == "v3_e2e_rnnt:1"
    assert loader.loads == [api.MODEL_NAME]


def test_cache_evicts_least_recently_used_model_and_releases_it():
    loader = Loader()
    with TestClient(api.create_app(model_loader=loader, model_cache_size=2)) as client:
        transcribe(client, "multilingual_ctc")
        transcribe(client)  # Default is now the most recently used model.
        transcribe(client, "multilingual_large_ctc")
        assert loader.references[1]() is None
        assert transcribe(client).json()["text"] == f"{api.MODEL_NAME}:1"
        assert (
            transcribe(client, "multilingual_ctc").json()["text"]
            == "multilingual_ctc:2"
        )
    assert all(ref() is None for ref in loader.references)


def test_concurrent_requests_share_one_model_load():
    loader = Loader()
    barrier = Barrier(4)
    with TestClient(api.create_app(model_loader=loader)) as client:

        def request(_):
            barrier.wait(timeout=5)
            return transcribe(client, "multilingual_large_ctc").json()

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(request, range(4)))
        assert (
            results
            == [{"text": "multilingual_large_ctc:1", "model": "multilingual_large_ctc"}]
            * 4
        )
    assert loader.loads == [api.MODEL_NAME, "multilingual_large_ctc"]


def test_failed_model_load_is_retryable():
    loader = Loader()
    failed = False

    def flaky_loader(name=api.MODEL_NAME):
        nonlocal failed
        if name == "multilingual_ctc" and not failed:
            failed = True
            raise RuntimeError("download unavailable")
        return loader(name)

    with TestClient(
        api.create_app(model_loader=flaky_loader), raise_server_exceptions=False
    ) as client:
        assert transcribe(client, "multilingual_ctc").status_code == 503
        assert transcribe(client).json()["text"] == f"{api.MODEL_NAME}:1"
        assert (
            transcribe(client, "multilingual_ctc").json()["text"]
            == "multilingual_ctc:1"
        )


def test_cache_limit_must_be_positive():
    with pytest.raises(ValueError):
        api.create_app(model_loader=Loader(), model_cache_size=0)


def test_single_model_cache_reloads_default_after_switching():
    loader = Loader()
    with TestClient(api.create_app(model_loader=loader, model_cache_size=1)) as client:
        assert (
            transcribe(client, "multilingual_ctc").json()["model"] == "multilingual_ctc"
        )
        assert loader.references[0]() is None
        assert transcribe(client).json()["text"] == f"{api.MODEL_NAME}:2"
        assert loader.references[1]() is None


def test_selected_model_supports_longform_and_word_timestamps():
    with TestClient(api.create_app(model_loader=Loader())) as client:
        for audio, extra in [(b"audio", {"words": []}), (b"long", {"segments": []})]:
            response = client.post(
                "/transcribe",
                params={"model": "multilingual_ctc", "word_timestamps": "true"},
                files={"file": ("voice.wav", audio)},
            )
            assert response.status_code == 200
            assert response.json() == {
                "text": "multilingual_ctc:1",
                "model": "multilingual_ctc",
                **extra,
            }


def test_default_alias_is_reported_as_canonical_model(monkeypatch):
    monkeypatch.setattr(api, "MODEL_NAME", "rnnt")
    loader = Loader()
    with TestClient(api.create_app(model_loader=loader)) as client:
        assert client.get("/health").json()["model"] == "v3_rnnt"
        assert transcribe(client).json() == {"text": "v3_rnnt:1", "model": "v3_rnnt"}
    assert loader.loads == ["v3_rnnt"]


def test_failed_inference_does_not_retain_evicted_model_in_traceback():
    references = []

    class InvalidAudioModel(Model):
        def transcribe(self, path, word_timestamps=False):
            raise RuntimeError("Failed to load audio")

    def loader(name):
        if name == api.MODEL_NAME:
            model = InvalidAudioModel(name, 1)
        else:
            assert references[0]() is None
            model = Model(name, 1)
        references.append(weakref.ref(model))
        return model

    app = api.create_app(model_loader=loader, model_cache_size=1)
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/transcribe"
    )
    with TestClient(app) as client:
        # Hold the exception as if its response is still being handled while
        # the next request switches models after inference releases the lock.
        with pytest.raises(HTTPException) as error:
            endpoint(
                file=UploadFile(file=BytesIO(b"invalid")),
                model=None,
                word_timestamps=False,
            )
        assert error.value.status_code == 422
        assert transcribe(client, "multilingual_ctc").status_code == 200


def test_failed_load_releases_partial_model_before_retry():
    partial_references = []

    def loader(name):
        model = Model(name, 1)
        if name == "multilingual_ctc":
            if not partial_references:
                partial_references.append(weakref.ref(model))
                raise RuntimeError("download unavailable")
            assert partial_references[0]() is None
        return model

    app = api.create_app(model_loader=loader)
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/transcribe"
    )
    with TestClient(app) as client:
        with pytest.raises(HTTPException) as error:
            endpoint(
                file=UploadFile(file=BytesIO(b"audio")),
                model="multilingual_ctc",
                word_timestamps=False,
            )
        assert error.value.status_code == 503
        assert transcribe(client, "multilingual_ctc").status_code == 200
