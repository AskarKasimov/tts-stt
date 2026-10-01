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
превышение размера — с 413. Текущая HTTP-обёртка принимает записи до 25 секунд.

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
