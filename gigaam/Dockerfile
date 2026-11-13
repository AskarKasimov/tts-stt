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
COPY gigaam/ .

ARG TARGETARCH

# The locked torchcodec release has Linux wheels only for x86_64.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETARCH" = "amd64" ]; then \
      uv sync --locked --no-dev --extra longform --extra api --no-editable; \
    else \
      uv sync --locked --no-dev --extra torch --extra api --no-editable; \
    fi

COPY receiver /opt/receiver

# Receiver dependencies are pinned separately from model dependencies.
RUN uv export --project /opt/receiver --frozen --no-dev --no-emit-project --format requirements-txt -o /tmp/receiver-requirements.txt \
    && uv pip install --python /app/.venv/bin/python --require-hashes -r /tmp/receiver-requirements.txt \
    && uv pip install --python /app/.venv/bin/python --no-deps /opt/receiver

EXPOSE 8000
CMD ["python", "-m", "speech_receiver.main", "--config", "/app/config/prod.yaml"]
