import io
import os
import re
import threading

import soundfile as sf
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

MODEL_ID = os.getenv("MODEL_ID", "openbmb/VoxCPM2")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_CHUNK_LENGTH = 500
CHUNK_PAUSE_SECONDS = 0.15
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


class LongformSynthesisRequest(SynthesisRequest):
    text: str = Field(min_length=1, max_length=10_000)


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


def _synthesize_longform(request: LongformSynthesisRequest) -> bytes:
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


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "loaded": _model is not None}


@app.post("/synthesize", response_class=Response)
async def synthesize(request: SynthesisRequest):
    audio = await run_in_threadpool(_synthesize, request)
    return Response(content=audio, media_type="audio/wav")


@app.post("/synthesize/longform", response_class=Response)
async def synthesize_longform(request: LongformSynthesisRequest):
    audio = await run_in_threadpool(_synthesize_longform, request)
    return Response(content=audio, media_type="audio/wav")
