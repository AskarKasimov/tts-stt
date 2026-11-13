#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up --build "$@"
