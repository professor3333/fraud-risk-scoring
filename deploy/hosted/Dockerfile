# The service for hosts that build from the public repository (Render, Koyeb, …;
# docs/deployment.md). Identical to the root Dockerfile except that the model is NOT in
# the image: at startup the service fetches models/champion/ from FRAUD_CHAMPION_URL
# (the champion-<sha> GitHub release of this repository; FRAUD_CHAMPION_TOKEN only for
# a private store) and runs its usual parity check before serving. Binds to $PORT, which these hosts set.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    FRAUD_AUDIT_DB=/app/audit/prediction_events.sqlite

WORKDIR /app
COPY pyproject.toml uv.lock LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group train --no-install-project
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group train

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/audit /app/models/champion \
    && chown -R app:app /app/audit /app/models
USER app

EXPOSE 8000
CMD ["/bin/sh", "-c", "exec /app/.venv/bin/uvicorn fraud.serve.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
