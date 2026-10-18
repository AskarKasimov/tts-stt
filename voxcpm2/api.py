import asyncio
import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from runtime import ReferenceFeatures, reference_cache_size
from streaming import StreamCancelled, StreamJob

MODEL_ID = os.getenv("MODEL_ID", "openbmb/VoxCPM2")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_CHUNK_LENGTH = 500
CHUNK_PAUSE_SECONDS = 0.15
STREAM_DRAIN_TIMEOUT = 5.0
REFERENCE_CACHE_DIR = Path(os.getenv("REFERENCE_CACHE_DIR", "/model-cache/voices"))
REFERENCE_TEXT = (
    "Давайте внимательно разберём задачу. Сначала определите, что уже известно "
    "и какой результат нужно получить. Затем выберите подходящий способ решения "
    "и объясните свой выбор. Если возникнут сомнения, вернитесь к условию, "
    "проверьте каждый шаг и спокойно продолжайте работу."
)
logger = logging.getLogger("uvicorn.error")
_model = None
_model_lock = threading.Lock()
_ready = False
_startup_started = False
_initialization_error = None
_features = ReferenceFeatures()


@asynccontextmanager
async def lifespan(app):
    _start_runtime()
    yield

app = FastAPI(title="VoxCPM2 TTS", lifespan=lifespan)
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
model_wait = Histogram("voxcpm_model_wait_seconds", "Time waiting for the model lock.", registry=metrics_registry)
reference_build = Histogram("voxcpm_reference_build_seconds", "Reference feature lookup/build time.", registry=metrics_registry)
generation_duration = Histogram("voxcpm_generation_seconds", "Time generating target speech.", registry=metrics_registry)
reference_cache = Counter("voxcpm_reference_cache_total", "Reference feature lookups.", ("outcome",), registry=metrics_registry)
badcase_retries = Counter("voxcpm_badcase_retries_total", "Target badcase retries.", registry=metrics_registry)
stream_events = Counter("voxcpm_streams_total", "Streaming synthesis outcomes.", ("outcome",), registry=metrics_registry)
stream_ttfa = Histogram("voxcpm_stream_ttfa_seconds", "Request to first queued PCM.", registry=metrics_registry)


@contextmanager
def model_lock():
    started = time.perf_counter()
    _model_lock.acquire()
    model_wait.observe(time.perf_counter() - started)
    try:
        yield
    finally:
        _model_lock.release()


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    cfg_value: float = Field(default=2.0, ge=0.0, le=5.0)
    inference_timesteps: int = Field(default=20, ge=1, le=50)
    seed: int | None = 510
    use_reference: bool = Field(
        default=True,
        description="Create or reuse a persistent reference voice for seed, CFG and inference steps.",
    )

    @model_validator(mode="after")
    def reference_requires_seed(self):
        if self.use_reference and self.seed is None:
            raise ValueError("seed must not be null when use_reference is true")
        return self

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
    """Load the eager model once under the model lock, without running synthesis."""
    global _model, _ready, _initialization_error, _features
    if _model is None:
        if _initialization_error is not None:
            raise RuntimeError("Model initialization failed; restart after fixing configuration") from _initialization_error
        try:
            size = reference_cache_size()
            from voxcpm import VoxCPM
            started = time.perf_counter()
            model = VoxCPM.from_pretrained(MODEL_ID, load_denoiser=False, optimize=False, device=DEVICE)
            logger.info("Model loaded: total_seconds=%.3f load_seconds=%s device=%s", time.perf_counter() - started, getattr(model, "load_seconds", None), DEVICE)
            _features = ReferenceFeatures(size)
            _model, _ready = model, True
        except Exception as error:
            _initialization_error = error
            _ready = False
            raise
    return _model


def _start_runtime():
    global _startup_started
    _startup_started = True
    def initialize():
        try:
            with model_lock():
                _load_model()
        except Exception:
            logger.exception("Runtime initialization failed; readiness remains unavailable")
    threading.Thread(target=initialize, name="voxcpm-initialize", daemon=True).start()


def _validate_audio(audio):
    audio = np.asarray(audio)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("Synthesis returned invalid audio")
    return audio


def _reference_for(model, request: SynthesisRequest, cancel_event=None) -> str | None:
    """Called under _model_lock; references survive model and container restarts."""
    if not request.use_reference:
        return None
    cache_key = json.dumps(
        {
            "version": 1,
            "model": MODEL_ID,
            "text": REFERENCE_TEXT,
            "seed": request.seed,
            "cfg_value": float(request.cfg_value) or 0.0,
            "inference_timesteps": request.inference_timesteps,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(cache_key.encode()).hexdigest()
    path = REFERENCE_CACHE_DIR / f"seed_{request.seed}_{digest}.wav"
    if path.exists():
        try:
            audio, sample_rate = sf.read(path, dtype="float32")
            if (
                audio.ndim == 1 and audio.size > 0 and np.isfinite(audio).all()
                and sample_rate == model.tts_model.sample_rate
            ):
                return str(path)
        except (RuntimeError, ValueError):
            pass
        logger.warning("Regenerating invalid reference: %s", path)

    audio = model.generate(
        text=REFERENCE_TEXT,
        cfg_value=request.cfg_value,
        inference_timesteps=request.inference_timesteps,
        seed=request.seed,
        retry_badcase=False,
        **({"cancel_event": cancel_event} if cancel_event is not None else {}),
    )
    audio = np.asarray(audio)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("Reference synthesis returned invalid audio")
    REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=REFERENCE_CACHE_DIR, suffix=".wav", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        sf.write(temp_path, audio, model.tts_model.sample_rate, format="WAV", subtype="PCM_16")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info("Created reference: %s", path)
    return str(path)


def _generation_options(model, reference):
    options = {}
    if reference is not None:
        if hasattr(model.tts_model, "build_prompt_cache"):
            started = time.perf_counter()
            prepared, hit = _features.get(model, reference, MODEL_ID)
            reference_build.observe(time.perf_counter() - started)
            reference_cache.labels(outcome="hit" if hit else "miss").inc()
            options["prompt_cache"] = prepared
        else:
            options["reference_wav_path"] = reference
    return options


def _generate_target(model, request: SynthesisRequest, text: str, reference: str | None):
    options = _generation_options(model, reference)
    started = time.perf_counter()
    audio = model.generate(
        text=text,
        cfg_value=request.cfg_value,
        inference_timesteps=request.inference_timesteps,
        seed=request.seed,
        **options,
    )
    generation_duration.observe(time.perf_counter() - started)
    badcase_retries.inc(getattr(model.tts_model, "last_badcase_retries", 0))
    logger.info(
        "Synthesized target: requested_seed=%s actual_seed=%s reference=%s",
        request.seed, getattr(model.tts_model, "last_successful_seed", None), reference,
    )
    return _validate_audio(audio)


def _synthesize(request: SynthesisRequest) -> bytes:
    with model_lock():
        model = _load_model()
        reference = _reference_for(model, request)
        audio = _generate_target(model, request, request.text, reference)
        output = io.BytesIO()
        sf.write(output, audio, model.tts_model.sample_rate, format="WAV")
        return output.getvalue()


def _synthesize_longform(request: SynthesisRequest) -> bytes:
    chunks = split_text(request.text)
    with model_lock():
        model = _load_model()
        reference = _reference_for(model, request)
        sample_rate = model.tts_model.sample_rate
        output = io.BytesIO()
        with sf.SoundFile(output, mode="w", samplerate=sample_rate, channels=1, format="WAV") as wav:
            for index, chunk in enumerate(chunks):
                audio = _generate_target(model, request, chunk, reference)
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


class SynthesisMiddleware:
    """ASGI middleware preserves per-event flush and bounds blocked transport sends."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or scope.get("method") != "POST" or path not in {"/synthesize", "/synthesize/stream"}:
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        status = 500
        async def measured_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            if path.endswith("/stream") and message["type"] == "http.response.body":
                await asyncio.wait_for(send(message), timeout=STREAM_DRAIN_TIMEOUT)
            else:
                await send(message)
        try:
            await self.app(scope, receive, measured_send)
        finally:
            job = scope.get("voxcpm_stream_job")
            if job is not None:
                job.cancel.set()
            if not (path.endswith("/stream") and 200 <= status < 300):
                outcome = "success" if 200 <= status < 300 else "client_error" if 400 <= status < 500 else "server_error"
                speech_requests.labels(outcome=outcome).inc()
                speech_request_duration.observe(time.perf_counter() - started)


app.add_middleware(SynthesisMiddleware)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "loaded": _model is not None}


@app.get("/ready")
def ready():
    return Response(content=json.dumps({"status": "ready" if _ready else "unavailable"}),
                    status_code=200 if _ready else 503, media_type="application/json")


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(metrics_registry), media_type=CONTENT_TYPE_LATEST)


@app.post("/synthesize", response_class=Response)
async def synthesize(request: SynthesisRequest):
    _require_available()
    generate = _synthesize if len(request.text) <= MAX_CHUNK_LENGTH else _synthesize_longform
    audio = await run_in_threadpool(generate, request)
    duration = wav_duration_seconds(audio)
    if duration is None:
        speech_audio_duration_unavailable.inc()
    else:
        speech_audio_seconds.labels(direction="output").inc(duration)
    speech_text_characters.inc(len(request.text))
    return Response(content=audio, media_type="audio/wav")


def _require_available():
    if _initialization_error is not None or (_startup_started and not _ready):
        raise HTTPException(status_code=503, detail="TTS runtime unavailable")


def _stream_job(request):
    def produce(job):
        locked = False
        try:
            started = time.perf_counter()
            while not locked:
                job.check()
                locked = _model_lock.acquire(timeout=0.05)
            model_wait.observe(time.perf_counter() - started)
            job.check()
            model = _load_model()
            job.check()
            job.sample_rate = model.tts_model.sample_rate
            if not isinstance(job.sample_rate, int) or not 8000 <= job.sample_rate <= 192000:
                raise RuntimeError("Invalid model sample rate")
            job.put({"type": "start", "version": 1, "sample_rate": job.sample_rate,
                     "channels": 1, "encoding": "pcm_s16le"})
            reference = _reference_for(model, request, cancel_event=job.cancel)
            chunks = split_text(request.text) if len(request.text) > MAX_CHUNK_LENGTH else [request.text]
            for index, text in enumerate(chunks):
                job.check()
                options = _generation_options(model, reference)
                generator = model.generate_streaming(text=text, cfg_value=request.cfg_value,
                    inference_timesteps=request.inference_timesteps, seed=request.seed,
                    retry_badcase=False, cancel_event=job.cancel, **options)
                started = time.perf_counter()
                try:
                    for waveform in generator:
                        job.check()
                        job.audio(waveform)
                finally:
                    generator.close()
                    generation_duration.observe(time.perf_counter() - started)
                if index < len(chunks) - 1:
                    job.audio(np.zeros(int(job.sample_rate * CHUNK_PAUSE_SECONDS), dtype=np.float32))
            if job.samples == 0:
                raise RuntimeError("Streaming synthesis returned no audio")
            # Transport drain and terminal event do not keep the model locked.
            _model_lock.release()
            locked = False
            job.put({"type": "end", "chunks": job.chunks, "samples": job.samples})
            job.outcome = "success"
        except StreamCancelled:
            job.outcome = "interrupted"
        except Exception:
            job.outcome = "interrupted" if job.cancel.is_set() else "error"
            if not job.cancel.is_set():
                logger.exception("Streaming synthesis failed")
            if locked:
                _model_lock.release()
                locked = False
            try:
                job.put({"type": "error", "code": "SYNTHESIS_FAILED"})
            except StreamCancelled:
                pass
        finally:
            if locked:
                _model_lock.release()
    return StreamJob(produce, timeout=STREAM_DRAIN_TIMEOUT)


@app.post("/synthesize/stream")
async def synthesize_stream(request: SynthesisRequest, http_request: Request):
    _require_available()
    try:
        reference_cache_size()
        if _initialization_error is not None:
            raise RuntimeError("Runtime initialization failed")
    except (ValueError, RuntimeError, OSError) as error:
        logger.error("Stream unavailable: %s", error)
        raise HTTPException(status_code=503, detail="TTS runtime unavailable") from error
    job = _stream_job(request)
    http_request.scope["voxcpm_stream_job"] = job

    async def body():
        completed = False
        try:
            async for event in job.body():
                yield event
                if event.startswith(b'{"type":"end"'):
                    completed = True
        finally:
            job.cancel.set()
            outcome = "success" if completed else "server_error"
            speech_requests.labels(outcome=outcome).inc()
            speech_request_duration.observe(time.perf_counter() - job.started)
            stream_events.labels(outcome="success" if completed else job.outcome if job.outcome != "success" else "interrupted").inc()
            if job.first_audio is not None:
                stream_ttfa.observe(job.first_audio - job.started)
            if completed:
                speech_audio_seconds.labels(direction="output").inc(job.samples / job.sample_rate)
                speech_text_characters.inc(len(request.text))
    return StreamingResponse(body(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
