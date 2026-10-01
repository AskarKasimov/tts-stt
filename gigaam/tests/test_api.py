"""HTTP contract tests that do not download model weights."""

from dataclasses import dataclass
import sys
from types import ModuleType

from fastapi.testclient import TestClient

import api
from api import create_app


@dataclass
class FakeResult:
    text: str


class FakeModel:
    def transcribe(self, path):
        with open(path, "rb") as audio:
            content = audio.read()
        if content == b"long":
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")
        if content == b"invalid":
            raise RuntimeError("Failed to load audio")
        return FakeResult("Привет, мир!")


def test_transcribe_returns_text():
    with TestClient(create_app(model_loader=FakeModel)) as client:
        response = client.post("/transcribe", files={"file": ("voice.ogg", b"audio")})
    assert response.status_code == 200
    assert response.json()["text"] == "Привет, мир!"


def test_empty_audio_is_rejected():
    with TestClient(create_app(model_loader=FakeModel)) as client:
        response = client.post("/transcribe", files={"file": ("voice.wav", b"")})
    assert response.status_code == 422


def test_oversized_audio_is_rejected():
    with TestClient(create_app(model_loader=FakeModel, max_upload_bytes=3)) as client:
        response = client.post("/transcribe", files={"file": ("voice.wav", b"audio")})
    assert response.status_code == 413


def test_invalid_and_long_audio_are_rejected():
    with TestClient(create_app(model_loader=FakeModel)) as client:
        invalid = client.post("/transcribe", files={"file": ("voice.wav", b"invalid")})
        long = client.post("/transcribe", files={"file": ("voice.wav", b"long")})
    assert invalid.status_code == 422
    assert long.status_code == 422


def test_model_loader_uses_configured_device(monkeypatch):
    fake_gigaam = ModuleType("gigaam")
    fake_gigaam.load_model = lambda name, device: (name, device)
    monkeypatch.setitem(sys.modules, "gigaam", fake_gigaam)
    monkeypatch.setattr(api, "DEVICE", "cuda")
    assert api.load_model() == (api.MODEL_NAME, "cuda")
