"""stdlib client and real audio round-trip check; also usable outside Docker."""

import argparse
import base64
import io
import json
import math
import struct
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

MODEL = "qwen3-omni-speech"
SYSTEM = (
    "Ты голосовой помощник. Отвечай по-русски в одном или двух коротких предложениях. "
    "Отвечай на вопрос пользователя. Пиши только то, что нужно произнести вслух."
)


def validate_wav(data, *, max_seconds=None):
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError("Нужен WAV PCM, 16 бит, моно. Команда преобразования есть в README.")
        frames, rate = wav.getnframes(), wav.getframerate()
        if rate <= 0 or frames <= 0:
            raise ValueError("Аудио пустое.")
        seconds = frames / rate
        if max_seconds is not None and seconds > max_seconds:
            raise ValueError(f"Аудио длиннее {max_seconds} секунд.")
        pcm = wav.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError("Аудио обрезано.")
        samples = [value[0] for value in struct.iter_unpack("<h", pcm)]
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        if rms < 2:
            raise ValueError("Аудио содержит тишину вместо речи.")
        return {"seconds": round(seconds, 3), "sample_rate": rate, "rms": round(rms, 2)}


def decode_reply(result):
    texts = []
    audio_data = None
    for choice in result.get("choices", []):
        message = choice.get("message", {})
        text = message.get("content")
        if isinstance(text, str) and text.strip():
            texts.append(text)
        audio = message.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            audio_data = base64.b64decode(audio["data"], validate=True)
    if not texts or audio_data is None:
        raise RuntimeError("Нужны одновременно текст и аудио; проверьте все три стадии Omni.")
    return "\n".join(texts), audio_data, validate_wav(audio_data)


def request_reply(base_url, content, output, *, timeout=900):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        "modalities": ["text", "audio"], "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read(4096).decode(errors='replace')}") from error
    text, audio_data, audio_info = decode_reply(result)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(audio_data)
    output.with_suffix(".txt").write_text(text, encoding="utf-8")
    report = {"text": text, "audio": audio_info, "elapsed_seconds": round(time.perf_counter() - started, 2)}
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(text, flush=True)
    print(f"Ответ сохранён: {output}; {report['elapsed_seconds']} с", flush=True)
    return report


def speech_reply(base_url, path, output, *, max_seconds=30):
    data = Path(path).read_bytes()
    if len(data) > 10 * 1024**2:
        raise ValueError("Лимит клиента — 10 MiB.")
    validate_wav(data, max_seconds=max_seconds)
    encoded = base64.b64encode(data).decode("ascii")
    return request_reply(base_url, [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + encoded}}], output)


def self_test(base_url, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The seed is synthesized by the model: no external demo URL or extra TTS dependency.
    seed = output_dir / "test-question.wav"
    request_reply(base_url, "Прочитай вслух ровно эту фразу: Какого цвета ясное дневное небо?", seed)
    speech_reply(base_url, seed, output_dir / "test-answer.wav", max_seconds=60)
    print("Техническая проверка TEXT → AUDIO → TEXT+AUDIO пройдена. Прослушайте test-answer.wav.")


def main():
    parser = argparse.ArgumentParser(description="Запрос Qwen3-Omni с сохранением WAV и текста.")
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=Path)
    group.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/answer.wav"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    if args.self_test:
        self_test(args.url, args.output_dir)
    else:
        speech_reply(args.url, args.audio, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"ОШИБКА: {error}") from error
