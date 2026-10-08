"""Workload metrics tests that do not download model weights."""

from dataclasses import dataclass
from math import nan
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

import api
from api import create_app


@dataclass
class FakeResult:
    text: str
    words: list


class FakeModel:
    def transcribe(self, path, word_timestamps=False):
        return FakeResult("recognized", [])


def sample_value(metrics: str, name: str, **labels: str) -> float:
    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            if sample.name == name and all(
                sample.labels.get(key) == value for key, value in labels.items()
            ):
                return sample.value
    return 0.0


def test_metrics_count_successful_transcribes_and_ignore_other_routes(monkeypatch):
    monkeypatch.setattr(api, "probe_audio_duration", lambda path: 12.5)
    with TestClient(create_app(model_loader=FakeModel)) as client:
        response = client.post("/transcribe", files={"file": ("voice.wav", b"audio")})
        assert response.status_code == 200
        client.get("/health")
        client.get("/metrics")
        client.get("/metrics")
        metrics = client.get("/metrics").text

    assert sample_value(metrics, "speech_requests_total", outcome="success") == 1
    assert sample_value(
        metrics, "speech_audio_seconds_total", direction="input"
    ) == 12.5
    assert sample_value(metrics, "speech_request_duration_seconds_count") == 1


def test_metrics_count_empty_upload_as_client_error_without_audio_seconds(monkeypatch):
    monkeypatch.setattr(api, "probe_audio_duration", lambda path: 12.5)
    with TestClient(create_app(model_loader=FakeModel)) as client:
        response = client.post("/transcribe", files={"file": ("voice.wav", b"")})
        metrics = client.get("/metrics").text

    assert response.status_code == 422
    assert sample_value(metrics, "speech_requests_total", outcome="client_error") == 1
    assert sample_value(
        metrics, "speech_audio_seconds_total", direction="input"
    ) == 0


@pytest.mark.parametrize("duration", [None, nan, -1])
def test_metrics_count_unavailable_audio_duration_without_changing_response(
    monkeypatch, duration
):
    monkeypatch.setattr(api, "probe_audio_duration", lambda path: duration)
    with TestClient(create_app(model_loader=FakeModel)) as client:
        response = client.post("/transcribe", files={"file": ("voice.wav", b"audio")})
        metrics = client.get("/metrics").text

    assert response.status_code == 200
    assert sample_value(metrics, "speech_audio_duration_unavailable_total") == 1
    assert sample_value(
        metrics, "speech_audio_seconds_total", direction="input"
    ) == 0


def test_metrics_probe_exception_does_not_change_transcription_response(monkeypatch):
    def broken_probe(path):
        raise OSError("ffprobe failed")

    monkeypatch.setattr(api, "probe_audio_duration", broken_probe)
    with TestClient(create_app(model_loader=FakeModel)) as client:
        response = client.post("/transcribe", files={"file": ("voice.wav", b"audio")})
        metrics = client.get("/metrics").text

    assert response.status_code == 200
    assert sample_value(metrics, "speech_audio_duration_unavailable_total") == 1


@pytest.mark.parametrize(
    "result,raises,expected",
    [
        (SimpleNamespace(stdout="12.5\n", returncode=0), None, 12.5),
        (None, FileNotFoundError(), None),
        (None, subprocess.TimeoutExpired("ffprobe", 5), None),
        (None, subprocess.CalledProcessError(1, "ffprobe"), None),
        (SimpleNamespace(stdout="oops", returncode=0), None, None),
        (SimpleNamespace(stdout="nan", returncode=0), None, None),
        (SimpleNamespace(stdout="-1", returncode=0), None, None),
    ],
)
def test_probe_audio_duration_handles_ffprobe_results(
    monkeypatch, result, raises, expected
):
    def run(*args, **kwargs):
        if raises:
            raise raises
        return result

    monkeypatch.setattr(api.subprocess, "run", run)
    assert api.probe_audio_duration(Path("audio.wav")) == expected
