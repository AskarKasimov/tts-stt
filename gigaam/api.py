"""HTTP adapter with automatic short-form and long-form speech recognition."""

import asyncio
import math
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response


MODEL_NAME = os.getenv("MODEL_NAME", "v3_e2e_rnnt")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def probe_audio_duration(path: Path) -> float | None:
    """Return decoded audio duration from ffprobe, or None when unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        duration = float(result.stdout.strip())
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
        UnicodeError,
    ):
        return None
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def load_model():
    import gigaam

    return gigaam.load_model(MODEL_NAME, device=DEVICE)


def create_app(
    model_loader: Callable = load_model, max_upload_bytes: int = MAX_UPLOAD_BYTES
) -> FastAPI:
    model_lock = Lock()
    registry = CollectorRegistry()
    request_count = Counter(
        "speech_requests_total",
        "Speech transcription requests by outcome.",
        ["outcome"],
        registry=registry,
    )
    request_duration = Histogram(
        "speech_request_duration_seconds",
        "Time spent handling speech transcription requests.",
        registry=registry,
    )
    audio_seconds = Counter(
        "speech_audio_seconds_total",
        "Audio duration processed by direction.",
        ["direction"],
        registry=registry,
    )
    duration_unavailable = Counter(
        "speech_audio_duration_unavailable_total",
        "Successful transcriptions whose input duration could not be measured.",
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = await asyncio.to_thread(model_loader)
        yield

    app = FastAPI(title="GigaAM ASR", lifespan=lifespan)

    @app.middleware("http")
    async def track_transcription_requests(request: Request, call_next):
        if request.method != "POST" or request.url.path != "/transcribe":
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
            request_count.labels(outcome=outcome).inc()
            request_duration.observe(time.perf_counter() - started)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}

    def transcribe_audio(file: UploadFile, word_timestamps: bool = False):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio"
            size = 0
            with audio_path.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(413, "Audio file is too large")
                    output.write(chunk)

            if size == 0:
                raise HTTPException(422, "Audio file is empty")

            longform = False
            try:
                with model_lock:
                    try:
                        result = app.state.model.transcribe(
                            str(audio_path), word_timestamps=word_timestamps
                        )
                    except ValueError as exc:
                        # GigaAM checks the decoded sample count before inference.
                        if str(exc) != (
                            "Too long wav file, use 'transcribe_longform' method."
                        ):
                            raise
                        longform = True
                        result = app.state.model.transcribe_longform(
                            str(audio_path), word_timestamps=word_timestamps
                        )
            except ImportError as exc:
                if longform:
                    raise HTTPException(
                        503,
                        "Long-form dependencies are unavailable; install the longform extra",
                    ) from exc
                raise
            except RuntimeError as exc:
                if str(exc) == "Failed to load audio":
                    raise HTTPException(422, "Unsupported or invalid audio") from exc
                if longform and "and no HF_TOKEN was provided" in str(exc):
                    raise HTTPException(
                        503,
                        "Long-form VAD weights are unavailable; "
                        "configure HF_TOKEN or cache the weights",
                    ) from exc
                raise
            response = {"text": result.text, "model": MODEL_NAME}
            if longform:
                response["segments"] = [asdict(segment) for segment in result.segments]
            elif word_timestamps:
                response["words"] = [asdict(word) for word in result.words or []]
            try:
                duration = probe_audio_duration(audio_path)
            except Exception:
                duration = None
            if duration is None or not math.isfinite(duration) or duration < 0:
                duration_unavailable.inc()
            else:
                audio_seconds.labels(direction="input").inc(duration)
            return response

    @app.post("/transcribe")
    def transcribe(
        file: UploadFile = File(...),
        word_timestamps: bool = Query(
            False,
            description="Include word timestamps in seconds from the start of the recording",
        ),
    ):
        return transcribe_audio(file, word_timestamps=word_timestamps)

    return app


app = create_app()
