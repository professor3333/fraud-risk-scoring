"""FastAPI inference service: the calibrated pipeline, loaded once, scoring raw rows (G8)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from fraud.data import schema
from fraud.evaluate.threshold import load_threshold_config
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.schemas import (
    IDENTITY_FIELDS,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
INPUT_COLUMNS: tuple[str, ...] = (
    tuple(c for c in schema.TRANSACTION_COLS if c != schema.TARGET_COL) + IDENTITY_FIELDS
)
DEFAULT_CONFIG = ROOT / "configs" / "serving.yaml"


@dataclass(frozen=True)
class ServingState:
    model: CalibratedModel
    threshold: float
    model_version: str


def load_state(config_path: Path = DEFAULT_CONFIG) -> ServingState:
    raw = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parents[1]
    model_path = root / raw["model_path"]
    model = joblib.load(model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    threshold = load_threshold_config(root / raw["threshold_config"]).threshold
    return ServingState(
        model=model, threshold=threshold, model_version=f"{raw['model_version']}@{digest}"
    )


def request_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """One validated request -> a one-row frame shaped like the training frame."""
    row = {c: payload.get(c) for c in INPUT_COLUMNS}
    row[schema.HAS_IDENTITY_COL] = any(row[c] is not None for c in IDENTITY_FIELDS)
    frame = pd.DataFrame([row])
    for col in schema.TRANSACTION_STR_COLS | schema.IDENTITY_STR_COLS:
        frame[col] = frame[col].astype("str")
    numeric = [
        c
        for c in frame.columns
        if c not in schema.TRANSACTION_STR_COLS | schema.IDENTITY_STR_COLS
        and c != schema.HAS_IDENTITY_COL
    ]
    frame[numeric] = frame[numeric].astype("float64")
    frame[schema.ID_COL] = frame[schema.ID_COL].astype("int64")
    frame[schema.TIME_COL] = frame[schema.TIME_COL].astype("int64")
    return frame


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
        return HealthResponse(status="ok", model_version=s.model_version, threshold=s.threshold)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: TransactionRequest, request: Request) -> PredictionResponse:  # type: ignore[valid-type]
        return score(request.app.state.serving, body.model_dump())  # type: ignore[attr-defined]

    return app


app = create_app()
