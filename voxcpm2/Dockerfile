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
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra api --no-editable

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
