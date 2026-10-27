"""Speech and text API; the model is served by the separate Omni process."""

import asyncio
import base64
import os
import tempfile
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import JSONResponse, Response

from client import MODEL, SYSTEM, decode_reply, validate_wav

MAX_FILE_BYTES = 10 * 1024**2
MAX_BODY_BYTES = MAX_FILE_BYTES + 1024**2
MAX_SECONDS = 30
MAX_CHAT_BODY_BYTES = 128 * 1024


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8000, strict=True)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.content.strip():
            raise ValueError("Текст сообщения не должен быть пустым.")
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(default=256, ge=1, le=1024, strict=True)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)

    @model_validator(mode="after")
    def bound_history(self):
        if sum(len(message.content) for message in self.messages) > 12000:
            raise ValueError("История превышает 12 000 символов.")
        return self


async def normalize_audio(data):
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input"
        target = Path(directory) / "normalized.wav"
        source.write_bytes(data)
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(source),
            "-t", "31", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(process.communicate(), timeout=20)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.communicate()
            raise
        if process.returncode or not target.exists():
            raise HTTPException(400, "Не удалось прочитать аудио. Поддерживаются WAV, MP3, OGG, WebM и FLAC.")
        normalized = target.read_bytes()
        try:
            with wave.open(str(target), "rb") as wav:
                seconds = wav.getnframes() / wav.getframerate()
            if seconds > MAX_SECONDS:
                raise HTTPException(413, f"Запись длиннее {MAX_SECONDS} секунд.")
            validate_wav(normalized, max_seconds=MAX_SECONDS)
        except (ValueError, wave.Error, EOFError) as error:
            raise HTTPException(400, str(error)) from error
        return normalized


class AdmissionMiddleware:
    """Share one generation slot across text/speech and bound request bodies."""

    def __init__(self, app, rejected):
        self.app, self.rejected = app, rejected
        self.busy = False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in self.rejected or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        counter = self.rejected[scope["path"]]
        limit = MAX_CHAT_BODY_BYTES if scope["path"] == "/chat" else MAX_BODY_BYTES
        if self.busy:
            counter.labels(outcome="busy").inc()
            return await JSONResponse({"detail": "Модель занята. Повторите запрос позже."}, 429,
                                      headers={"Retry-After": "5"})(scope, receive, send)
        self.busy = True
        try:
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > limit:
                    counter.labels(outcome="too_large").inc()
                    return await JSONResponse({"detail": "Превышен лимит размера запроса."}, 413)(scope, receive, send)
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            replayed = False

            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
        finally:
            self.busy = False


def create_app(*, transport=None, normalizer=normalize_audio):
    registry = CollectorRegistry()
    requests = Counter("speech_requests_total", "Speech requests by outcome", ["outcome"], registry=registry)
    duration = Histogram("speech_request_duration_seconds", "Speech request time", registry=registry)
    text_requests = Counter("text_requests_total", "Text requests by outcome", ["outcome"], registry=registry)
    text_duration = Histogram("text_request_duration_seconds", "Text request time", registry=registry)
    engine_url = os.getenv("OMNI_URL", "http://omni:8091").rstrip("/")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(900, connect=5)) as client:
            app.state.engine = client
            yield

    app = FastAPI(title="Qwen3-Omni Speech-to-Speech", lifespan=lifespan)
    app.add_middleware(AdmissionMiddleware, rejected={"/speech-to-speech": requests, "/chat": text_requests})

    @app.get("/health")
    async def health():
        try:
            response = await app.state.engine.get(engine_url + "/health", timeout=5)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(503, "Движок модели не готов.") from error
        return {"status": "ok", "model": MODEL, "quantization": "int4-AutoRound"}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/chat")
    async def chat(body: ChatRequest):
        started, outcome = time.perf_counter(), "error"
        try:
            payload = {
                "model": MODEL, "modalities": ["text"], "stream": False,
                "messages": [message.model_dump() for message in body.messages],
                "max_tokens": body.max_tokens, "temperature": body.temperature,
            }
            try:
                response = await app.state.engine.post(engine_url + "/v1/chat/completions", json=payload)
                response.raise_for_status()
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Пустой текстовый ответ.")
            except httpx.TimeoutException as error:
                raise HTTPException(504, "Истёк таймаут ответа модели.") from error
            except httpx.ConnectError as error:
                raise HTTPException(503, "Нет соединения с движком модели.") from error
            except httpx.HTTPStatusError as error:
                if error.response.status_code in (400, 422):
                    raise HTTPException(422, "Модель отклонила запрос. Сократите историю или max_tokens.") from error
                raise HTTPException(502, "Ошибка текстового движка. Проверьте логи Omni.") from error
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
                raise HTTPException(502, "Движок не вернул корректный текстовый ответ.") from error
            outcome = "success"
            answer = {"text": text, "model": MODEL}
            if isinstance(result.get("usage"), dict):
                answer["usage"] = result["usage"]
            return answer
        finally:
            text_requests.labels(outcome=outcome).inc()
            text_duration.observe(time.perf_counter() - started)

    @app.post("/speech-to-speech", responses={200: {"content": {"audio/wav": {}}}})
    async def speech_to_speech(
        file: UploadFile = File(...),
        response_format: Literal["wav", "json"] = Query("wav"),
    ):
        started, outcome = time.perf_counter(), "error"
        try:
            data = await file.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, "Лимит файла — 10 MiB.")
            if not data:
                raise HTTPException(400, "Пустой файл.")
            try:
                normalized = await normalizer(data)
            except asyncio.TimeoutError as error:
                raise HTTPException(422, "Декодирование аудио превысило 20 секунд.") from error
            encoded = base64.b64encode(normalized).decode("ascii")
            payload = {
                "model": MODEL, "modalities": ["text", "audio"], "stream": False,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + encoded}}
                ]}],
            }
            try:
                response = await app.state.engine.post(engine_url + "/v1/chat/completions", json=payload)
                response.raise_for_status()
                text, audio, audio_info = decode_reply(response.json())
            except httpx.TimeoutException as error:
                raise HTTPException(504, "Истёк таймаут ответа модели.") from error
            except httpx.ConnectError as error:
                raise HTTPException(503, "Нет соединения с движком модели.") from error
            except (httpx.HTTPError, ValueError, RuntimeError, wave.Error, EOFError, KeyError, TypeError) as error:
                raise HTTPException(502, "Движок не вернул корректные текст и WAV. Проверьте логи Omni.") from error
            outcome = "success"
            if response_format == "json":
                return {"text": text, "model": MODEL, "audio": {
                    "format": "wav", "sample_rate": audio_info["sample_rate"],
                    "data": base64.b64encode(audio).decode("ascii"),
                }}
            return Response(audio, media_type="audio/wav", headers={"Content-Disposition": 'attachment; filename="answer.wav"'})
        finally:
            await file.close()
            requests.labels(outcome=outcome).inc()
            duration.observe(time.perf_counter() - started)

    return app


app = create_app()
