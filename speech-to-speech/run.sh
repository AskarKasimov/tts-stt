#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
mkdir -p data output reports

dc() { docker compose -f compose.yaml "$@"; }

diagnostics() {
  dc ps --all > reports/containers.txt 2>&1 || true
  dc logs --no-color --tail 300 omni > reports/omni.log 2>&1 || true
  dc logs --no-color --tail 100 api > reports/api.log 2>&1 || true
  nvidia-smi > reports/gpu.txt 2>&1 || true
}

failed() {
  local result=$?
  trap - ERR
  diagnostics
  printf '\nОперация завершилась с ошибкой (%s). Диагностика: speech-to-speech/reports/\n' "$result" >&2
  printf 'Посмотрите причину выше и reports/omni.log. Контейнер не перезапускается автоматически.\n' >&2
  exit "$result"
}
trap failed ERR

host_check() { python3 host_check.py; }
gpu_check() { dc run --rm --no-deps --pull never omni check; }
prepare() { dc run --rm --no-deps --pull never download; }
start() { dc up -d --no-build --pull never --wait --wait-timeout 1800 omni api; }
verify() {
  dc exec -T omni python3 /opt/s2s/client.py --self-test --output-dir /output
  dc exec -T omni python3 /opt/s2s/verify_api.py
}

case "${1:-help}" in
  check) host_check ;;
  build) dc build --pull omni ;;
  gpu-check) gpu_check ;;
  prepare) prepare ;;
  start) start ;;
  verify) verify ;;
  install)
    host_check
    dc build --pull omni
    gpu_check
    prepare
    start
    verify
    diagnostics
    printf '\nСервис запущен, техническая проверка аудио пройдена. Прослушайте output/test-answer.wav.\n'
    ;;
  deploy)
    host_check
    gpu_check
    start
    verify
    diagnostics
    ;;
  status) dc ps --all ;;
  logs) dc logs -f --tail 100 omni api ;;
  stop) dc down ;;
  doctor) diagnostics; printf 'Диагностика сохранена в reports/\n' ;;
  test)
    dc run --rm --no-deps --entrypoint python3 -v "$PWD:/opt/s2s:ro" download -m unittest discover -s /opt/s2s/tests -v
    # Separate project: never connects a fake engine to the production service.
    test_result=0
    docker compose -f compose.cpu-test.yaml up --abort-on-container-exit --exit-code-from probe || test_result=$?
    docker compose -f compose.cpu-test.yaml down
    exit "$test_result"
    ;;
  *)
    printf 'Команды: check | install | build | gpu-check | prepare | deploy | start | verify | status | logs | stop | doctor | test\n'
    ;;
esac
