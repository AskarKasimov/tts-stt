"""Isolated real-network HTTP test. The fake is never used in production Compose."""

import base64
import io
import json
import struct
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx


class Fake(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] == "qwen3-omni-speech"
        if body["modalities"] == ["text"]:
            assert body["messages"][-1]["content"] == "Сколько будет два плюс два?"
            result = {"choices": [{"message": {"content": "Четыре."}}]}
        else:
            assert body["modalities"] == ["text", "audio"]
            url = body["messages"][1]["content"][0]["audio_url"]["url"]
            result = {"choices": [{"message": {"content": "Ответ заглушки."}}, {"message": {
                "audio": {"data": url.split(",", 1)[1]}
            }}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())


def probe():
    with httpx.Client(base_url="http://api:8000", timeout=30) as client:
        for attempt in range(100):
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("API did not become ready")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack("<h", 1200) * 1600)
        data = buffer.getvalue()
        response = client.post("/speech-to-speech", files={"file": ("test.wav", data, "audio/wav")})
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            assert wav.getnframes() == 1600
        response = client.post("/speech-to-speech?response_format=json", files={"file": ("test.wav", data)})
        assert response.status_code == 200, response.text
        assert response.json()["text"] == "Ответ заглушки."
        assert base64.b64decode(response.json()["audio"]["data"]).startswith(b"RIFF")
        assert client.post("/speech-to-speech", files={"file": ("bad", b"broken")}).status_code == 400
        assert "speech_requests_total" in client.get("/metrics").text
        assert client.get("/docs").status_code == 200
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "Сколько будет два плюс два?"}]})
        assert response.status_code == 200, response.text
        assert response.json() == {"text": "Четыре.", "model": "qwen3-omni-speech"}
        assert "text_requests_total" in client.get("/metrics").text
        print("PASS: real Uvicorn + speech/text routes + ffmpeg + fake upstream + WAV/JSON + errors/metrics/docs; no model inference.")


if __name__ == "__main__":
    if sys.argv[1] == "fake":
        HTTPServer(("0.0.0.0", 8091), Fake).serve_forever()
    else:
        probe()
