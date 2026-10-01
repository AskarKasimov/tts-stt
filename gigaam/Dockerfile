FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra torch --extra api --no-editable

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
