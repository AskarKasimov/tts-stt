"""Run after the direct-model self test, through the real multipart API."""

from pathlib import Path

import httpx

from client import validate_wav


def main():
    response = httpx.post("http://api:8000/chat", json={
        "messages": [{"role": "user", "content": "Сколько будет два плюс два? Ответь кратко."}],
        "max_tokens": 64, "temperature": 0,
    }, timeout=900)
    response.raise_for_status()
    answer = response.json()
    if not isinstance(answer.get("text"), str) or not answer["text"].strip() or "audio" in answer:
        raise RuntimeError("Текстовая ручка вернула некорректный ответ.")
    Path("/output/api-chat-answer.txt").write_text(answer["text"], encoding="utf-8")
    print("HTTP /chat: текстовый ответ сохранён в output/api-chat-answer.txt.")
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
