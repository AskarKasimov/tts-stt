"""Multipart speech API; the model is served by the separate Omni process."""

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
from starlette.responses import JSONResponse, Response

from client import MODEL, SYSTEM, decode_reply, validate_wav

MAX_FILE_BYTES = 10 * 1024**2
MAX_BODY_BYTES = MAX_FILE_BYTES + 1024**2
MAX_SECONDS = 30


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
    """Admit one generation and limit bytes before the multipart parser runs."""

    def __init__(self, app, rejected):
        self.app, self.rejected = app, rejected
        self.busy = False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/speech-to-speech" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        if self.busy:
            self.rejected.labels(outcome="busy").inc()
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
                if len(body) + len(chunk) > MAX_BODY_BYTES:
                    self.rejected.labels(outcome="too_large").inc()
                    return await JSONResponse({"detail": "Слишком большой запрос; лимит файла 10 MiB."}, 413)(scope, receive, send)
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
    engine_url = os.getenv("OMNI_URL", "http://omni:8091").rstrip("/")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(900, connect=5)) as client:
            app.state.engine = client
            yield

    app = FastAPI(title="Qwen3-Omni Speech-to-Speech", lifespan=lifespan)
    app.add_middleware(AdmissionMiddleware, rejected=requests)

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
