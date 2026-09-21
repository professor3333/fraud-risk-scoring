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
# ONE uvicorn worker, deliberately, and no knob to change it. The review budget and the
# per-client rate limiter live in this process — its memory and its SQLite audit trail —
# so a second worker would charge a different budget and hand the same transaction a
# different action (docs/model_card.md: "The features are stateless; the service is not").
# Scaling this service means sharing that state first, not adding processes.
CMD ["/bin/sh", "-c", "exec /app/.venv/bin/uvicorn fraud.serve.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
