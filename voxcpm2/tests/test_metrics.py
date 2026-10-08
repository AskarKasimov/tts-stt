import numpy as np
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

import api


def sample_value(metrics: str, name: str, **labels: str) -> float:
    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            if sample.name == name and all(
                sample.labels.get(key) == value for key, value in labels.items()
            ):
                return sample.value
    return 0.0


def current(client: TestClient) -> str:
    return client.get("/metrics").text


def test_synthesis_metrics_track_success_validation_longform_and_unavailable(monkeypatch):
    class FakeModel:
        tts_model = type("TTS", (), {"sample_rate": 48000})()

        def generate(self, **kwargs):
            return np.zeros(480, dtype=np.float32)

    monkeypatch.setattr(api, "_load_model", FakeModel)
    client = TestClient(api.app)

    before = current(client)
    before_success = sample_value(before, "speech_requests_total", outcome="success")
    before_audio = sample_value(
        before, "speech_audio_seconds_total", direction="output"
    )
    before_chars = sample_value(before, "speech_text_characters_total")
    response = client.post("/synthesize", json={"text": "Привет"})
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"
    after = current(client)
    assert sample_value(after, "speech_requests_total", outcome="success") - before_success == 1
    assert sample_value(after, "speech_audio_seconds_total", direction="output") - before_audio == 0.01
    assert sample_value(after, "speech_text_characters_total") - before_chars == len("Привет")

    before = after
    response = client.post("/synthesize", json={"text": ""})
    assert response.status_code == 422
    after = current(client)
    assert sample_value(after, "speech_requests_total", outcome="client_error") - sample_value(
        before, "speech_requests_total", outcome="client_error"
    ) == 1
    assert sample_value(after, "speech_audio_seconds_total", direction="output") == sample_value(
        before, "speech_audio_seconds_total", direction="output"
    )
    assert sample_value(after, "speech_text_characters_total") == sample_value(
        before, "speech_text_characters_total"
    )

    text = "А" * 490 + ". " + "Б" * 20
    before = after
    response = client.post("/synthesize", json={"text": text})
    assert response.status_code == 200
    after = current(client)
    assert sample_value(after, "speech_requests_total", outcome="success") - sample_value(
        before, "speech_requests_total", outcome="success"
    ) == 1
    duration_delta = sample_value(
        after, "speech_audio_seconds_total", direction="output"
    ) - sample_value(before, "speech_audio_seconds_total", direction="output")
    assert duration_delta == (480 + 7200 + 480) / 48000
    assert sample_value(after, "speech_text_characters_total") - sample_value(
        before, "speech_text_characters_total"
    ) == len(text)

    monkeypatch.setattr(api, "wav_duration_seconds", lambda audio: None)
    before = after
    response = client.post("/synthesize", json={"text": "duration unavailable"})
    assert response.status_code == 200
    after = current(client)
    assert sample_value(after, "speech_audio_duration_unavailable_total") - sample_value(
        before, "speech_audio_duration_unavailable_total"
    ) == 1


def test_health_and_metrics_do_not_count_as_synthesis_requests():
    client = TestClient(api.app)
    before = current(client)
    client.get("/health")
    client.get("/metrics")
    after = current(client)
    for outcome in ("success", "client_error", "server_error"):
        assert sample_value(after, "speech_requests_total", outcome=outcome) == sample_value(
            before, "speech_requests_total", outcome=outcome
        )
