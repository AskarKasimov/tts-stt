import asyncio
import base64
import io
import json
import struct
import sys
import unittest
import wave
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import MAX_BODY_BYTES, MAX_FILE_BYTES, create_app, normalize_audio


def audio(seconds=0.1):
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<h", 1200) * int(seconds * 16000))
    return data.getvalue()


def reply(data=None):
    return {"choices": [{"message": {"content": "Четыре."}}, {"message": {"audio": {
        "data": base64.b64encode(data or audio()).decode()
    }}}]}


class ApiTests(unittest.TestCase):
    def client(self, handler):
        client = TestClient(create_app(transport=httpx.MockTransport(handler)))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_multipart_wav_normalization_and_reply(self):
        seen = []

        def engine(request):
            payload = json.loads(request.content)
            seen.append(payload)
            return httpx.Response(200, json=reply())

        response = self.client(engine).post("/speech-to-speech", files={"file": ("question.wav", audio())})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, audio())
        self.assertEqual(seen[0]["modalities"], ["text", "audio"])
        self.assertTrue(seen[0]["messages"][1]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,"))

    def test_json_reply_includes_text_and_audio(self):
        client = self.client(lambda request: httpx.Response(200, json=reply()))
        response = client.post("/speech-to-speech?response_format=json", files={"file": ("q.wav", audio())})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Четыре.")
        self.assertEqual(base64.b64decode(response.json()["audio"]["data"]), audio())

    def test_text_without_voice_is_502(self):
        client = self.client(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "Текст"}}]}))
        self.assertEqual(client.post("/speech-to-speech", files={"file": ("q.wav", audio())}).status_code, 502)

    def test_invalid_audio_is_400_without_calling_model(self):
        def engine(request):
            self.fail("Не следует вызывать модель для повреждённого аудио")

        response = self.client(engine).post("/speech-to-speech", files={"file": ("bad.wav", b"not audio")})
        self.assertEqual(response.status_code, 400)

    def test_too_long_audio_is_413(self):
        client = self.client(lambda request: self.fail("Не следует вызывать модель"))
        response = client.post("/speech-to-speech", files={"file": ("long.wav", audio(30.5))})
        self.assertEqual(response.status_code, 413)

    def test_large_file_is_413(self):
        client = self.client(lambda request: self.fail("Не следует вызывать модель"))
        response = client.post("/speech-to-speech", files={"file": ("big.wav", bytes(MAX_FILE_BYTES + 1))})
        self.assertEqual(response.status_code, 413)

    def test_oversized_body_is_rejected_before_multipart(self):
        client = self.client(lambda request: self.fail("Не следует вызывать модель"))
        response = client.post("/speech-to-speech", content=bytes(MAX_BODY_BYTES + 1))
        self.assertEqual(response.status_code, 413)

    def test_health_reflects_engine_readiness(self):
        client = self.client(lambda request: httpx.Response(503))
        self.assertEqual(client.get("/health").status_code, 503)

    def test_timeout_is_504_and_health_remains_accessible(self):
        def engine(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            raise httpx.ReadTimeout("timeout", request=request)

        client = self.client(engine)
        self.assertEqual(client.post("/speech-to-speech", files={"file": ("q.wav", audio())}).status_code, 504)
        self.assertEqual(client.get("/health").status_code, 200)

    def test_metrics_include_success(self):
        client = self.client(lambda request: httpx.Response(200, json=reply()))
        client.post("/speech-to-speech", files={"file": ("q.wav", audio())})
        self.assertIn('speech_requests_total{outcome="success"} 1.0', client.get("/metrics").text)

    def test_busy_admission_and_release(self):
        async def check():
            entered, release = asyncio.Event(), asyncio.Event()

            async def normalizer(data):
                entered.set()
                await release.wait()
                return data

            app = create_app(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply())), normalizer=normalizer)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    first = asyncio.create_task(client.post("/speech-to-speech", files={"file": ("q.wav", audio())}))
                    await asyncio.wait_for(entered.wait(), 5)
                    second = await client.post("/speech-to-speech", files={"file": ("q.wav", audio())})
                    self.assertEqual(second.status_code, 429)
                    self.assertEqual((await client.get("/health")).status_code, 200)
                    release.set()
                    self.assertEqual((await first).status_code, 200)
                    self.assertEqual((await client.post("/speech-to-speech", files={"file": ("q.wav", audio())})).status_code, 200)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
