"""HTTP adapter for GigaAM short-form and long-form speech recognition."""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel


MODEL_NAME = os.getenv("MODEL_NAME", "v3_e2e_rnnt")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class WordTimestamp(BaseModel):
    text: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    words: list[WordTimestamp] | None = None


class LongformTranscription(BaseModel):
    text: str
    model: str
    segments: list[TranscriptSegment]


def load_model():
    import gigaam

    return gigaam.load_model(MODEL_NAME, device=DEVICE)


def create_app(
    model_loader: Callable = load_model, max_upload_bytes: int = MAX_UPLOAD_BYTES
) -> FastAPI:
    model_lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = await asyncio.to_thread(model_loader)
        yield

    app = FastAPI(title="GigaAM ASR", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}

    def transcribe_audio(
        file: UploadFile, longform: bool = False, word_timestamps: bool = False
    ):
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

            try:
                with model_lock:
                    if longform:
                        result = app.state.model.transcribe_longform(
                            str(audio_path), word_timestamps=word_timestamps
                        )
                    else:
                        result = app.state.model.transcribe(str(audio_path))
            except ImportError as exc:
                if longform:
                    raise HTTPException(
                        503,
                        "Long-form dependencies are unavailable; install the longform extra",
                    ) from exc
                raise
            except ValueError as exc:
                if "Too long wav file" in str(exc):
                    raise HTTPException(422, "Audio must be at most 25 seconds") from exc
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
            return response

    @app.post("/transcribe")
    def transcribe(file: UploadFile = File(...)):
        return transcribe_audio(file)

    @app.post("/transcribe/longform", response_model=LongformTranscription)
    def transcribe_longform(
        file: UploadFile = File(...),
        word_timestamps: bool = Query(
            False,
            description="Include word timestamps in seconds from the start of the recording",
        ),
    ):
        return transcribe_audio(file, longform=True, word_timestamps=word_timestamps)

    return app


app = create_app()
