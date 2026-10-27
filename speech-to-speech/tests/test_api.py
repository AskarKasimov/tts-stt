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
from api import MAX_BODY_BYTES, MAX_CHAT_BODY_BYTES, MAX_FILE_BYTES, create_app


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

    def test_chat_preserves_history_and_requests_only_text(self):
        seen = []

        def engine(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "Четыре."}}],
                                           "usage": {"total_tokens": 20}})

        messages = [{"role": "system", "content": "Отвечай развёрнуто."},
                    {"role": "user", "content": "Два плюс два?"}]
        response = self.client(engine).post("/chat", json={"messages": messages, "max_tokens": 512, "temperature": 0})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "Четыре.", "model": "qwen3-omni-speech", "usage": {"total_tokens": 20}})
        self.assertEqual(seen[0], {"model": "qwen3-omni-speech", "modalities": ["text"], "stream": False,
                                  "messages": messages, "max_tokens": 512, "temperature": 0})

    def test_chat_does_not_decode_audio_or_require_audio_output(self):
        async def no_audio(data):
            self.fail("Текстовый запрос не должен вызывать аудиообработку")

        app = create_app(normalizer=no_audio, transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "Ответ"}}]})
        ))
        with TestClient(app) as client:
            response = client.post("/chat", json={"messages": [{"role": "user", "content": "Привет"}]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["text"], "Ответ")
            self.assertNotIn("audio", response.json())
            self.assertIn('text_requests_total{outcome="success"} 1.0', client.get("/metrics").text)

    def test_chat_rejects_invalid_or_multimodal_input(self):
        client = self.client(lambda request: self.fail("Некорректный ввод не должен попадать в модель"))
        for body in [
            {"messages": []},
            {"messages": [{"role": "user", "content": "   "}]},
            {"messages": [{"role": "user", "content": [{"type": "audio_url"}]}]},
            {"messages": [{"role": "tool", "content": "text"}]},
            {"messages": [{"role": "user", "content": "text"}], "max_tokens": 1025},
            {"messages": [{"role": "user", "content": "text"}], "modalities": ["audio"]},
            {"messages": [{"role": "user", "content": "a" * 7000}, {"role": "user", "content": "b" * 7000}]},
        ]:
            with self.subTest(body=str(body)[:80]):
                self.assertEqual(client.post("/chat", json=body).status_code, 422)

    def test_oversized_chat_body_is_413(self):
        client = self.client(lambda request: self.fail("Запрос не должен попадать в модель"))
        self.assertEqual(client.post("/chat", content=bytes(MAX_CHAT_BODY_BYTES + 1)).status_code, 413)

    def test_chat_invalid_upstream_response_is_502(self):
        for result in [{"choices": []}, {"choices": [{"message": {"content": ""}}]}, {"choices": None}]:
            with self.subTest(result=result):
                client = self.client(lambda request: httpx.Response(200, json=result))
                self.assertEqual(client.post("/chat", json={"messages": [{"role": "user", "content": "Тест"}]}).status_code, 502)

    def test_chat_engine_errors(self):
        body = {"messages": [{"role": "user", "content": "Тест"}]}
        for status, expected in [(400, 422), (422, 422), (500, 502)]:
            with self.subTest(status=status):
                client = self.client(lambda request: httpx.Response(status))
                self.assertEqual(client.post("/chat", json=body).status_code, expected)
        for error_type, expected in [(httpx.ConnectError, 503), (httpx.ReadTimeout, 504)]:
            def engine(request):
                raise error_type("test", request=request)

            with self.subTest(error_type=error_type):
                self.assertEqual(self.client(engine).post("/chat", json=body).status_code, expected)

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
                    self.assertEqual((await client.post("/chat", json={"messages": [{"role": "user", "content": "Тест"}]})).status_code, 429)
                    self.assertEqual((await client.get("/health")).status_code, 200)
                    release.set()
                    self.assertEqual((await first).status_code, 200)
                    self.assertEqual((await client.post("/speech-to-speech", files={"file": ("q.wav", audio())})).status_code, 200)

        asyncio.run(check())

    def test_chat_and_speech_share_the_same_generation_slot(self):
        async def check():
            entered, release = asyncio.Event(), asyncio.Event()

            async def engine(request):
                entered.set()
                await release.wait()
                return httpx.Response(200, json={"choices": [{"message": {"content": "Ответ"}}]})

            app = create_app(transport=httpx.MockTransport(engine))
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    first = asyncio.create_task(client.post("/chat", json={"messages": [{"role": "user", "content": "Тест"}]}))
                    await asyncio.wait_for(entered.wait(), 5)
                    self.assertEqual((await client.post("/speech-to-speech", files={"file": ("q.wav", audio())})).status_code, 429)
                    release.set()
                    self.assertEqual((await first).status_code, 200)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
