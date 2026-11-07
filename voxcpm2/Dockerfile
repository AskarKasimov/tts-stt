FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    SETUPTOOLS_SCM_PRETEND_VERSION=2.0.3 \
    HF_HOME=/model-cache \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY voxcpm2/ .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra api --no-editable

COPY receiver /opt/receiver

# Receiver dependencies are pinned separately from model dependencies.
RUN uv export --project /opt/receiver --frozen --no-dev --no-emit-project --format requirements-txt -o /tmp/receiver-requirements.txt \
    && uv pip install --python /app/.venv/bin/python --require-hashes -r /tmp/receiver-requirements.txt \
    && uv pip install --python /app/.venv/bin/python --no-deps /opt/receiver

EXPOSE 8000
CMD ["python", "-m", "speech_receiver.main", "--config", "/app/config/prod.yaml"]
