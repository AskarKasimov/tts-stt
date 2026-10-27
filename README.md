# tts-stt

Сервисы распознавания, синтеза и диалоговой обработки речи:
* **GigaAM** (STT): распознавание речи (порт `8000`)
* **VoxCPM2** (TTS): синтез речи (порт `8001`)
* **Qwen3-Omni INT4** (Speech-to-Speech): сквозной голосовой диалог (порт `8002`)
* **Grafana / Prometheus**: мониторинг и метрики (порт `18473`)

Контракты репетитора находятся в [ai-tutor](https://github.com/AskarKasimov/ai-tutor).

---

## 1. Запуск на сервере с NVIDIA GPU

### GigaAM (STT) + VoxCPM2 (TTS)
Запуск основных сервисов STT и TTS:
```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d --build
```
* **STT Healthcheck:** `curl http://localhost:8000/health`
* **TTS Healthcheck:** `curl http://localhost:8001/health`
* **Grafana:** `http://<адрес-сервера>:18473` (логин/пароль настраиваются в `.env`)

### Qwen3-Omni (Speech-to-Speech)
Запускается **независимым** compose-файлом из корня (не затрагивает STT/TTS):
```bash
# 1. Скачать веса модели 26.3 ГБ (разово при первом запуске):
docker compose -f docker-compose.qwen.yaml run --rm download

# 2. Запустить сервисы:
docker compose -f docker-compose.qwen.yaml up -d --build
```
> *Рекомендуемый способ первого запуска с автоматической проверкой хоста, VRAM и аудио-тестом:*  
> `cd speech-to-speech && ./run.sh install`

* **Qwen3-Omni Healthcheck:** `curl http://localhost:8002/health`
* **Внутренний движок Omni:** `http://127.0.0.1:8091`

---

## 2. Локальный запуск на CPU (macOS / разработка)

Для локального тестирования STT и TTS без GPU:
```bash
docker compose up -d --build
```

---

## 3. Быстрая проверка API

### Распознавание (GigaAM)
```bash
curl -X POST 'http://localhost:8000/transcribe?word_timestamps=true' \
  -F 'file=@audio.wav'
```

### Синтез (VoxCPM2)
```bash
curl -X POST http://localhost:8001/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привет, мир!"}' --output speech.wav
```

### Голосовой диалог и чат (Qwen3-Omni)
```bash
# Speech-to-Speech (вход: аудио -> ответ: аудио WAV)
curl -X POST http://localhost:8002/speech-to-speech \
  -F 'file=@audio.wav' \
  -o reply.wav

# Текстовый чат
curl -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Привет!"}]}'
```

---

## 4. Подробная документация

* **GigaAM (STT):** [gigaam/README.md](gigaam/README.md)
* **VoxCPM2 (TTS):** [voxcpm2/API.md](voxcpm2/API.md)
* **Qwen3-Omni (Speech-to-Speech):**
  * Пошаговое развёртывание: [speech-to-speech/README.md](speech-to-speech/README.md)
  * Спецификация HTTP API: [speech-to-speech/API.md](speech-to-speech/API.md)
  * Эксплуатация и траблшутинг: [speech-to-speech/OPERATIONS.md](speech-to-speech/OPERATIONS.md)
