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
Qwen работает на той же GPU 80 ГБ рядом с GigaAM и VoxCPM2 (требует $\ge 60$ ГБ свободной VRAM перед стартом).

#### Первый запуск с нуля («под ключ»)
Перед запуском обязательно сбросьте кэш видеопамяти у соседних сервисов и запустите единый скрипт установки (проверит сервер, драйвер и GPU, скачает 26,3 ГБ весов, запустит контейнеры и проверит синтез аудио):

```bash
# 1. Обязательно сбросить кэш VRAM у работающих STT/TTS:
docker compose restart gigaam voxcpm2

# 2. Установка, скачивание весов и запуск с проверкой (рекомендуется в tmux):
cd speech-to-speech && ./run.sh install
```
> Команда завершится сообщением: `Сервис запущен, техническая проверка аудио пройдена. Прослушайте output/test-answer.wav`.

#### Последующий запуск и управление (из корня репозитория)
Когда веса уже скачаны, сервисом можно управлять прямо из корня через независимый Compose:
```bash
docker compose -f docker-compose.qwen.yaml up -d
docker compose -f docker-compose.qwen.yaml ps
docker compose -f docker-compose.qwen.yaml logs -f
docker compose -f docker-compose.qwen.yaml down
```

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
