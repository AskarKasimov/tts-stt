import io
import os
import threading

import soundfile as sf
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

MODEL_ID = os.getenv("MODEL_ID", "openbmb/VoxCPM2")
DEVICE = os.getenv("DEVICE", "cpu")
_model = None
_model_lock = threading.Lock()

app = FastAPI(title="VoxCPM2 TTS")


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
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


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "loaded": _model is not None}


@app.post("/synthesize", response_class=Response)
async def synthesize(request: SynthesisRequest):
    audio = await run_in_threadpool(_synthesize, request)
    return Response(content=audio, media_type="audio/wav")
