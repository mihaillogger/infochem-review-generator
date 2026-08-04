FROM python:3.11.15-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

FROM python:3.11.15-slim-bookworm

WORKDIR /app

ARG FASTEMBED_CACHE_DIR=/tmp/fastembed_cache
ARG HF_CACHE_DIR=/tmp/huggingface_cache

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /data/pdfs /data/images /data/processed ${FASTEMBED_CACHE_DIR} ${HF_CACHE_DIR} && \
    chown -R appuser:appuser /app /data ${FASTEMBED_CACHE_DIR} ${HF_CACHE_DIR}

COPY --chown=appuser:appuser --from=builder /app/.venv /app/.venv

COPY --chown=appuser:appuser src/ src/

USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src
ENV API_PORT=8000

EXPOSE $API_PORT

CMD ["sh", "-c", "uvicorn rag_core.main:app --host 0.0.0.0 --port $API_PORT"]