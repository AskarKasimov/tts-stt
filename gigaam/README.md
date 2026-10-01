# GigaAM HTTP API

Исходники инференса [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM),
исходный коммит `7447938d791c4f3e643386ee22c33777004293a5`.
Лицензия: [MIT](LICENSE). Сравнение моделей: [evaluation.md](evaluation.md).

## Запуск

Из корня репозитория `tts-stt`:

```bash
docker compose up -d --build gigaam
curl http://localhost:8000/health
curl -X POST http://localhost:8000/transcribe -F 'file=@voice.wav'
```

`POST /transcribe` принимает multipart-поле `file` до 25 МиБ и возвращает JSON
с `text` и `model`. Пустая или некорректная запись отклоняется с 422,
превышение размера — с 413. Эта ручка принимает записи до 25 секунд.

## Длинные записи

`POST /transcribe/longform` принимает то же multipart-поле `file` до 25 МиБ,
но не ограничивает запись 25 секундами. GigaAM выделяет речь с помощью VAD,
распознаёт фрагменты и возвращает общий текст и сегменты. Отдельного лимита
длительности нет; время обработки и память зависят от записи. Запрос синхронный.

```bash
curl -X POST 'http://localhost:8000/transcribe/longform?word_timestamps=true' \
  -F 'file=@long-answer.wav'
```

Пример ответа (таймкоды в секундах от начала исходной записи):

```json
{
  "text": "Привет!",
  "model": "v3_e2e_rnnt",
  "segments": [
    {
      "text": "Привет!",
      "start": 30.0,
      "end": 31.0,
      "words": [{"text": "Привет!", "start": 30.0, "end": 30.5}]
    }
  ]
}
```

Без `word_timestamps=true` поле `words` равно `null`. Если речь не обнаружена,
ответ содержит пустой `text` и `segments: []`.
Пустые и некорректные файлы возвращают 422; превышение размера — 413.
Отсутствующие зависимости long-form или некешированные VAD-веса без токена — 503.

В Docker для Linux x86_64 зависимости long-form включены. На Linux ARM64
образ поддерживает только короткие записи: у закреплённого `torchcodec==0.10.0`
нет подходящего Linux ARM64 wheel. Для нативного запуска на macOS ARM64
установите зависимости через `uv sync --extra api --extra longform --extra tests`.

Для первой загрузки VAD-модели `pyannote/segmentation-3.0` нужен `HF_TOKEN`
с доступом к модели. Задайте его в окружении перед запуском Compose:

```bash
export HF_TOKEN='<токен с доступом к VAD-модели>'
docker compose up -d --build gigaam
```

Веса VAD кешируются в volume `gigaam-vad-cache`. При наличии кеша токен
для повторной загрузки не требуется. Обычный `/transcribe` VAD не использует.

`MODEL_NAME` по умолчанию `v3_e2e_rnnt`, `DEVICE` — `cpu`.
Модель загружается при старте; веса кешируются в Docker volume.
Для NVIDIA GPU используйте [инструкцию в корне](../README.md).

## Проверка API без весов

Из этого каталога:

```bash
uv sync --extra api --extra tests
uv run --no-sync python -m pytest -q tests/test_api.py
```

В репозитории оставлены библиотека инференса и её тесты. Обучение,
Colab, Triton-развёртывание и иллюстрации удалены.
