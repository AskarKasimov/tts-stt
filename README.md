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
Модель по умолчанию задаётся через `MODEL_NAME` в Compose. Для отдельного
запроса её можно выбрать через `POST /transcribe?model=multilingual_large_ctc`
без перезапуска сервиса. Загруженные модели кешируются в RAM/VRAM
(до `MODEL_CACHE_SIZE`, по умолчанию 2), веса сохраняются в Docker volume.
Детали и допустимые имена — в [документации API](gigaam/README.md#выбор-модели-и-кеш).
Backend ai-tutor передаёт STT URL целиком, поэтому для выбора модели можно
задать `BACKEND_STT_URL=http://<адрес-сервера>:8000/transcribe?model=multilingual_large_ctc`.
Изменение настройки backend требует её применения на стороне backend;
повторный редеплой `tts-stt` для переключения модели не нужен.
Исходный код GigaAM взят из
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

При старте сервис загружает веса в Docker volume и прогревает WAV и streaming.
`/health` проверяет процесс, `/ready` возвращает 200 только после прогрева;
до этого синтез возвращает 503. В Docker Desktop на macOS
сервис работает на CPU; синтез может быть медленным.

По умолчанию `seed=510`, `cfg_value=2`, `inference_timesteps=20`,
`use_reference=true`: при первом обращении к комбинации параметров сервис
создаёт референс на фиксированной фразе и сохраняет WAV в volume. Последующие
запросы используют этот голос. Профиль 510/2/20 прогревается при старте.
Для стабильного голоса при сравнении 20/10 задайте `VOXCPM_REFERENCE_WAV` на
проверенную копию существующего WAV и запишите SHA-256; сервис не подменяет
этот файл при изменении параметров целевой реплики. `use_reference=false` отключает подготовку и
использование референса. Подробности — в [документации](voxcpm2/API.md#постоянный-голос).

`VoxCPM2` — актуальная модель в линейке VoxCPM с поддержкой русского языка.
Она выбрана ради качества и возможностей синтеза, а не скорости CPU-инференса.
`POST /synthesize` принимает до 10 000 символов: до 500 символов сервис
использует обычную генерацию, длиннее — озвучивает текст по частям и возвращает
один WAV. Подробности — в
[документации VoxCPM2](voxcpm2/API.md#длинный-текст).

`POST /synthesize/stream` принимает тот же JSON и отдаёт NDJSON с mono PCM16
сразу по мере генерации. Формат, ограничения, отмена и метрики описаны в
[API](voxcpm2/API.md#потоковый-синтез). WAV-путь сохранён. Для opt-in compile
есть `VOXCPM_OPTIMIZE=off|auto|on` (default `off`); CPU поддерживает `off` и
`auto`, `on` требует CUDA. `VOXCPM_REFERENCE_CACHE_SIZE=4` задаёт bounded LRU
признаков голоса (0–16, 0 отключает). Число шагов остаётся 20: переход на 10
требует новых CUDA-замеров и слепого прослушивания, включая вопросительную интонацию.

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

## Мониторинг нагрузки STT/TTS

Grafana и Prometheus запускаются вместе с основным Compose-проектом. Эти
контейнеры не получают GPU. Prometheus не публикуется наружу, а Grafana
доступна на всех сетевых интерфейсах сервера.

```bash
cp .env.example .env
# замените GRAFANA_ADMIN_PASSWORD в .env на настоящий секрет
docker compose up -d --build
docker compose ps
```

Grafana доступна извне по `http://<адрес-сервера>:18473`, если порт 18473 открыт
в межсетевом экране. Для доступа через интернет настройте HTTPS на reverse proxy
и создайте для провайдера пользователя с ролью Viewer. Цели Prometheus
проверяются на `http://localhost:9090/targets` из
самого контейнера. Дашборд показывает запросы, минуты входного STT-аудио,
минуты готового WAV TTS, задержки, ошибки и доступность сервисов.

История хранится до 90 дней в volumes `prometheus-data` и `grafana-data`.
Сбор начинается после развёртывания. Счётчики отражают HTTP-работу моделей,
включая тестовые запросы, и не являются числом пользователей, занятий или
GPU-часов.

Проверка endpoints после запуска:

```bash
curl http://localhost:8000/metrics
curl http://localhost:8001/metrics
```
