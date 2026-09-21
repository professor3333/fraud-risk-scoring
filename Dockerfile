# Build:  docker build -t fraud-risk-scoring .
# Run:    docker run --rm -p 8000:8000 -v fraud-audit:/app/audit fraud-risk-scoring
# The champion directory (models/champion/: artifact, frozen sample, manifest) must
# exist locally before building: scripts/promote.py writes it (docs/promotion.md).
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, from the lock file, so the layer caches across code changes.
COPY pyproject.toml uv.lock LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group train --no-install-project

# Project code, configs and the served artifact.
COPY src ./src
COPY configs ./configs
# The champion and its frozen sample: the service refuses to start unless the
# artifact reproduces the frozen probabilities (G8) and matches its manifest.
COPY models/champion/ ./models/champion/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group train

# Prediction audit trail: a writable directory for the service user; mount a
# volume at /app/audit to keep events across container restarts.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/audit && chown app:app /app/audit
ENV FRAUD_AUDIT_DB=/app/audit/prediction_events.sqlite
VOLUME ["/app/audit"]
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD ["/app/.venv/bin/python", "-c", \
    "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"]
# ONE uvicorn worker, deliberately, and no knob to change it. The review budget and the
# per-client rate limiter live in this process — its memory and its SQLite audit trail —
# so a second worker would charge a different budget and hand the same transaction a
# different action (docs/model_card.md: "The features are stateless; the service is not").
# Scaling this service means sharing that state first, not adding processes.
CMD ["/app/.venv/bin/uvicorn", "fraud.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
