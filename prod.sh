#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
for service in gigaam voxcpm2; do
    test -f "$service/config/prod.yaml" || { printf 'Missing %s/config/prod.yaml\n' "$service" >&2; exit 1; }
done
# Build images in CI or explicitly before deployment; do not compile on startup.
exec docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml -f docker-compose.prod.yaml up --no-build "$@"
