"""Run after the direct-model self test, through the real multipart API."""

from pathlib import Path

import httpx

from client import validate_wav


def main():
    question = Path("/output/test-question.wav").read_bytes()
    response = httpx.post(
        "http://api:8000/speech-to-speech", files={"file": ("question.wav", question, "audio/wav")}, timeout=900,
    )
    response.raise_for_status()
    validate_wav(response.content)
    Path("/output/api-answer.wav").write_bytes(response.content)
    print("HTTP multipart API: реальный WAV-ответ сохранён в output/api-answer.wav.")


if __name__ == "__main__":
    main()
