# Build:  docker build -t fraud-risk-scoring .
# Run:    docker run --rm -p 8000:8000 fraud-risk-scoring
# The model artifact (models/xgb_f5_capacity_calibrated.joblib) and its frozen
# sample must exist locally before building: scripts/train.py, scripts/calibrate.py,
# scripts/freeze_artifact.py.
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, from the lock file, so the layer caches across code changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group train --no-install-project

# Project code, configs and the served artifact.
COPY src ./src
COPY configs ./configs
# The artifact and its frozen sample: the service refuses to start unless the
# artifact reproduces the frozen probabilities (G8).
COPY models/xgb_f5_capacity_calibrated.joblib \
     models/xgb_f5_capacity_calibrated_frozen_sample.json \
     models/xgb_f5_capacity_calibrated_frozen_expected.json \
     ./models/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group train

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD ["/app/.venv/bin/python", "-c", \
    "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"]
CMD ["/app/.venv/bin/uvicorn", "fraud.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
