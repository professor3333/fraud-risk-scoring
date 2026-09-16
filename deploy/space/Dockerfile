# Hugging Face Space (Docker SDK) for the fraud risk scoring service — docs/deployment.md.
#
# The Space repository holds only these three files. The code is cloned from GitHub at
# FRAUD_REF (a Space *variable*; deploy.yml sets it to the release tag), and the model is
# NOT in the image: at startup the service fetches models/champion/ from a private HF
# model repo (FRAUD_CHAMPION_URL, a Space variable) with HF_TOKEN (a Space *secret*),
# then runs its usual parity check before serving. Nothing here contains the weights.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG FRAUD_REPO=https://github.com/professor3333/fraud-risk-scoring.git
ARG FRAUD_REF=main
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    FRAUD_AUDIT_DB=/app/audit/prediction_events.sqlite

WORKDIR /app
RUN git clone --depth 1 --branch "$FRAUD_REF" "$FRAUD_REPO" /app \
    && rm -rf /app/.git \
    && uv sync --frozen --no-dev --no-group train

# Spaces run as uid 1000; the champion is fetched into models/champion at startup and the
# audit trail is ephemeral on the free tier (no persistent volume) — both documented.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/audit /app/models/champion \
    && chown -R app:app /app/audit /app/models
USER app

EXPOSE 7860
CMD ["/app/.venv/bin/uvicorn", "fraud.serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
