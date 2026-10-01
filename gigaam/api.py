"""Small HTTP adapter for the upstream GigaAM short-form ASR model."""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Callable

from fastapi import FastAPI, File, HTTPException, UploadFile


MODEL_NAME = os.getenv("MODEL_NAME", "v3_e2e_rnnt")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


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

    @app.post("/transcribe")
    def transcribe(file: UploadFile = File(...)):
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
                    result = app.state.model.transcribe(str(audio_path))
            except ValueError as exc:
                if "Too long wav file" in str(exc):
                    raise HTTPException(422, "Audio must be at most 25 seconds") from exc
                raise
            except RuntimeError as exc:
                if str(exc) == "Failed to load audio":
                    raise HTTPException(422, "Unsupported or invalid audio") from exc
                raise
            return {"text": result.text, "model": MODEL_NAME}

    return app


app = create_app()
