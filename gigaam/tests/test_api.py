"""HTTP contract tests that do not download model weights."""

from dataclasses import dataclass
import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

import api
from api import create_app


@dataclass
class FakeResult:
    text: str


@dataclass
class FakeWord:
    text: str
    start: float
    end: float


@dataclass
class FakeSegment:
    text: str
    start: float
    end: float
    words: list[FakeWord] | None = None


@dataclass
class FakeLongformResult:
    segments: list[FakeSegment]

    @property
    def text(self):
        return " ".join(segment.text for segment in self.segments)


class FakeModel:
    def transcribe(self, path):
        with open(path, "rb") as audio:
            content = audio.read()
        if content == b"long":
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")
        if content == b"invalid":
            raise RuntimeError("Failed to load audio")
        return FakeResult("Привет, мир!")

    def transcribe_longform(self, path, word_timestamps=False):
        with open(path, "rb") as audio:
            content = audio.read()
        if content == b"invalid":
            raise RuntimeError("Failed to load audio")
        if content == b"silence":
            return FakeLongformResult([])
        words = [FakeWord("Привет!", 30.0, 30.5)] if word_timestamps else None
        return FakeLongformResult([FakeSegment("Привет!", 30.0, 31.0, words)])


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


def test_longform_accepts_audio_rejected_by_short_form():
    with TestClient(create_app(model_loader=FakeModel)) as client:
        short = client.post("/transcribe", files={"file": ("long.wav", b"long")})
        response = client.post(
            "/transcribe/longform", files={"file": ("long.wav", b"long")}
        )
    assert short.status_code == 422
    assert response.status_code == 200
    assert response.json() == {
        "text": "Привет!",
        "model": api.MODEL_NAME,
        "segments": [{"text": "Привет!", "start": 30.0, "end": 31.0, "words": None}],
    }


def test_longform_returns_word_timestamps_when_requested():
    with TestClient(create_app(model_loader=FakeModel)) as client:
        response = client.post(
            "/transcribe/longform?word_timestamps=true",
            files={"file": ("long.wav", b"long")},
        )
    assert response.status_code == 200
    assert response.json()["segments"][0]["words"] == [
        {"text": "Привет!", "start": 30.0, "end": 30.5}
    ]


@pytest.mark.parametrize("content", [b"", b"invalid"])
def test_longform_rejects_empty_and_invalid_audio(content):
    with TestClient(create_app(model_loader=FakeModel)) as client:
        response = client.post(
            "/transcribe/longform", files={"file": ("voice.wav", content)}
        )
    assert response.status_code == 422


def test_longform_rejects_oversized_audio():
    with TestClient(create_app(model_loader=FakeModel, max_upload_bytes=3)) as client:
        response = client.post(
            "/transcribe/longform", files={"file": ("long.wav", b"long")}
        )
    assert response.status_code == 413


def test_longform_returns_empty_result_for_silence():
    with TestClient(create_app(model_loader=FakeModel)) as client:
        response = client.post(
            "/transcribe/longform", files={"file": ("voice.wav", b"silence")}
        )
    assert response.status_code == 200
    assert response.json() == {"text": "", "model": api.MODEL_NAME, "segments": []}


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'pyannote'"),
        RuntimeError(
            "Model pyannote/segmentation-3.0 was not found locally, "
            "and no HF_TOKEN was provided to download it."
        ),
    ],
)
def test_longform_reports_unavailable_dependencies_or_vad_weights(error):
    class UnavailableLongformModel(FakeModel):
        def transcribe_longform(self, path, word_timestamps=False):
            raise error

    with TestClient(create_app(model_loader=UnavailableLongformModel)) as client:
        response = client.post(
            "/transcribe/longform", files={"file": ("long.wav", b"long")}
        )
        short = client.post("/transcribe", files={"file": ("voice.wav", b"audio")})
    assert response.status_code == 503
    assert short.status_code == 200
