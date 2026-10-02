# tts-stt

HTTP API распознавания и синтеза речи: GigaAM и VoxCPM2.
Репозиторий содержит исходники моделей, HTTP-обёртки, тесты и Docker-конфигурацию.
Развёртывание на отдельном сервере выполняет серверная команда.

Документация и контракты репетитора находятся в
[ai-tutor](https://github.com/AskarKasimov/ai-tutor), локально — `../ai-tutor`.

## GigaAM

Исходный код модели, HTTP API, Dockerfile и `uv.lock` находятся в
[`gigaam`](gigaam/README.md). Все команды `uv` для GigaAM
запускайте из этого каталога.

Запуск HTTP API из корня этого репозитория:

```bash
docker compose up -d --build gigaam
```

Проверка: `curl http://localhost:8000/health`.

`POST /transcribe` автоматически выбирает обработку по длительности: до
25 секунд — обычное распознавание, длиннее — VAD и longform:

```bash
curl -X POST 'http://localhost:8000/transcribe?word_timestamps=true' \
  -F 'file=@long-answer.wav'
```

Лимит файла — 25 МиБ; ответ содержит `text` и `model`, для длинных записей
также `segments`. По запросу возвращаются таймкоды слов.
Подготовка VAD-весов (`HF_TOKEN`) и поддерживаемые платформы описаны в
[документации GigaAM](gigaam/README.md#длинные-записи).

По умолчанию выбран `v3_e2e_rnnt`: он возвращает текст с пунктуацией и
нормализацией, что удобно для диалога с учеником. Если важнее минимальная
ошибка распознавания без оформления текста, в [оценке авторов](gigaam/evaluation.md)
`v3_rnnt` показывает средний WER 8,3% против 11,2% у `v3_e2e_rnnt`.
Модель меняется через `MODEL_NAME` в Compose. Исходный код GigaAM взят из
[официального репозитория](https://github.com/salute-developers/GigaAM), коммит
`7447938d791c4f3e643386ee22c33777004293a5`.

## VoxCPM2

Официальный исходный код [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
с HTTP API находится в [`voxcpm2`](voxcpm2/API.md). Используется исходный
коммит `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`.

```bash
docker compose up -d --build voxcpm2
curl http://localhost:8001/health
curl -X POST http://localhost:8001/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привет, мир!"}' --output speech.wav
```

Первый запрос скачает веса модели в Docker volume. В Docker Desktop на macOS
сервис работает на CPU; синтез может быть медленным.

`VoxCPM2` — актуальная модель в линейке VoxCPM с поддержкой русского языка.
Она выбрана ради качества и возможностей синтеза, а не скорости CPU-инференса.
`POST /synthesize` принимает до 10 000 символов: до 500 символов сервис
использует обычную генерацию, длиннее — озвучивает текст по частям и возвращает
один WAV. Подробности — в
[документации VoxCPM2](voxcpm2/API.md#длинный-текст).

## Запуск на сервере с NVIDIA GPU

На Linux x86_64 образы устанавливают PyTorch с CUDA 12.8. Для передачи GPU
обоим контейнерам нужны драйвер NVIDIA и NVIDIA Container Toolkit на сервере.
Запуск из корня этого репозитория:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d --build
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml ps
```

GPU-файл включает `DEVICE=cuda` и доступ к видеокартам для обоих сервисов.
Порты 8000 и 8001 опубликованы на всех сетевых интерфейсах сервера. На macOS
обычный `docker compose up -d --build` сохраняет CPU-режим.
