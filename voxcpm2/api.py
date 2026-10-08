import io
import math
import os
import re
import threading
import time

import soundfile as sf
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

MODEL_ID = os.getenv("MODEL_ID", "openbmb/VoxCPM2")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_CHUNK_LENGTH = 500
CHUNK_PAUSE_SECONDS = 0.15
_model = None
_model_lock = threading.Lock()

app = FastAPI(title="VoxCPM2 TTS")
metrics_registry = CollectorRegistry()
speech_requests = Counter(
    "speech_requests_total",
    "Speech synthesis requests by outcome.",
    ("outcome",),
    registry=metrics_registry,
)
speech_request_duration = Histogram(
    "speech_request_duration_seconds",
    "Speech synthesis request duration in seconds.",
    registry=metrics_registry,
)
speech_audio_seconds = Counter(
    "speech_audio_seconds_total",
    "Audio duration produced or consumed in seconds.",
    ("direction",),
    registry=metrics_registry,
)
speech_audio_duration_unavailable = Counter(
    "speech_audio_duration_unavailable_total",
    "Audio duration extractions that could not be completed.",
    registry=metrics_registry,
)
speech_text_characters = Counter(
    "speech_text_characters_total",
    "Characters submitted for speech synthesis.",
    registry=metrics_registry,
)


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    cfg_value: float = Field(default=2.0, ge=0.0, le=5.0)
    inference_timesteps: int = Field(default=10, ge=1, le=50)
    seed: int | None = None

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


def split_text(text: str) -> list[str]:
    remaining = re.sub(r"\s+", " ", text.strip())
    chunks = []
    while len(remaining) > MAX_CHUNK_LENGTH:
        window = remaining[:MAX_CHUNK_LENGTH]
        sentence_breaks = list(re.finditer(r'[.!?…]+[»”"\')\]]*(?=\s)', window))
        if sentence_breaks:
            cut = sentence_breaks[-1].end()
        else:
            cut = window.rfind(" ")
            if cut <= 0:
                cut = MAX_CHUNK_LENGTH
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _load_model():
    global _model
    if _model is None:
        from voxcpm import VoxCPM

        _model = VoxCPM.from_pretrained(
            MODEL_ID, load_denoiser=False, optimize=False, device=DEVICE
        )
    return _model


def _synthesize(request: SynthesisRequest) -> bytes:
    with _model_lock:
        model = _load_model()
        audio = model.generate(
            text=request.text,
            cfg_value=request.cfg_value,
            inference_timesteps=request.inference_timesteps,
            seed=request.seed,
        )
        output = io.BytesIO()
        sf.write(output, audio, model.tts_model.sample_rate, format="WAV")
        return output.getvalue()


def _synthesize_longform(request: SynthesisRequest) -> bytes:
    chunks = split_text(request.text)
    with _model_lock:
        model = _load_model()
        sample_rate = model.tts_model.sample_rate
        output = io.BytesIO()
        with sf.SoundFile(output, mode="w", samplerate=sample_rate, channels=1, format="WAV") as wav:
            for index, chunk in enumerate(chunks):
                audio = model.generate(
                    text=chunk,
                    cfg_value=request.cfg_value,
                    inference_timesteps=request.inference_timesteps,
                    seed=request.seed,
                )
                wav.write(audio)
                if index < len(chunks) - 1:
                    wav.write([0.0] * int(sample_rate * CHUNK_PAUSE_SECONDS))
        return output.getvalue()


def wav_duration_seconds(audio: bytes) -> float | None:
    try:
        info = sf.info(io.BytesIO(audio))
        seconds = info.frames / info.samplerate
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except (RuntimeError, ValueError, ZeroDivisionError):
        return None


@app.middleware("http")
async def measure_synthesis_requests(request, call_next):
    if request.method != "POST" or request.url.path != "/synthesize":
        return await call_next(request)

    started = time.perf_counter()
    outcome = "server_error"
    try:
        response = await call_next(request)
        if 200 <= response.status_code < 300:
            outcome = "success"
        elif 400 <= response.status_code < 500:
            outcome = "client_error"
        return response
    finally:
        speech_requests.labels(outcome=outcome).inc()
        speech_request_duration.observe(time.perf_counter() - started)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "loaded": _model is not None}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(metrics_registry), media_type=CONTENT_TYPE_LATEST)


@app.post("/synthesize", response_class=Response)
async def synthesize(request: SynthesisRequest):
    generate = _synthesize if len(request.text) <= MAX_CHUNK_LENGTH else _synthesize_longform
    audio = await run_in_threadpool(generate, request)
    duration = wav_duration_seconds(audio)
    if duration is None:
        speech_audio_duration_unavailable.inc()
    else:
        speech_audio_seconds.labels(direction="output").inc(duration)
    speech_text_characters.inc(len(request.text))
    return Response(content=audio, media_type="audio/wav")
