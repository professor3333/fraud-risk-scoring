"""FastAPI inference service: the calibrated pipeline, loaded once, scoring raw rows (G8)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from fraud.data import schema
from fraud.evaluate.threshold import load_threshold_config
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.frames import request_to_frame
from fraud.serve.parity import frozen_paths, verify
from fraud.serve.schemas import HealthResponse, PredictionResponse, TransactionRequest

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_CONFIG = ROOT / "configs" / "serving.yaml"


@dataclass(frozen=True)
class ServingState:
    model: CalibratedModel
    threshold: float
    model_version: str
    parity_rows: int


def load_state(config_path: Path = DEFAULT_CONFIG) -> ServingState:
    raw = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parents[1]
    model_path = root / raw["model_path"]
    model = joblib.load(model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    threshold = load_threshold_config(root / raw["threshold_config"]).threshold
    # Refuse to serve an artifact that does not reproduce its frozen probabilities (G8).
    if raw.get("require_parity", True):
        parity = verify(model, model_path)
        parity_rows = parity.n_rows
    else:
        parity_rows = 0 if not all(p.exists() for p in frozen_paths(model_path)) else -1
    return ServingState(
        model=model,
        threshold=threshold,
        model_version=f"{raw['model_version']}@{digest}",
        parity_rows=parity_rows,
    )


def score(state: ServingState, payload: dict[str, Any]) -> PredictionResponse:
    frame = request_to_frame(payload)
    p = float(state.model.predict_proba(frame)[0, 1])
    return PredictionResponse(
        transaction_id=int(payload[schema.ID_COL]),
        fraud_probability=p,
        decision="decline" if p >= state.threshold else "approve",
        threshold=state.threshold,
        model_version=state.model_version,
    )


def create_app(config_path: Path = DEFAULT_CONFIG) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.serving = load_state(config_path)
        yield

    app = FastAPI(title="fraud-risk-scoring", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        s: ServingState = request.app.state.serving
        return HealthResponse(
            status="ok",
            model_version=s.model_version,
            threshold=s.threshold,
            parity_rows=s.parity_rows,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: TransactionRequest, request: Request) -> PredictionResponse:  # type: ignore[valid-type]
        return score(request.app.state.serving, body.model_dump())  # type: ignore[attr-defined]

    return app


app = create_app()
