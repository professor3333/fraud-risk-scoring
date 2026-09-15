"""FastAPI inference service: the calibrated pipeline, loaded once, scoring raw rows (G8)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from fraud.data import schema
from fraud.evaluate.threshold import load_threshold_config
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.frames import payloads_to_frame, request_to_frame
from fraud.serve.parity import frozen_paths, verify
from fraud.serve.schemas import (
    Action,
    Bands,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RankedPrediction,
    RiskLevel,
    TransactionRequest,
)

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_CONFIG = ROOT / "configs" / "serving.yaml"


@dataclass(frozen=True)
class ServingState:
    model: CalibratedModel
    threshold: float
    bands: Bands
    model_version: str
    parity_rows: int
    info: dict[str, Any]


def load_state(config_path: Path = DEFAULT_CONFIG) -> ServingState:
    raw = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parents[1]
    model_path = root / raw["model_path"]
    model = joblib.load(model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    threshold = load_threshold_config(root / raw["threshold_config"]).threshold
    bands = Bands(**raw.get("bands", {"review": threshold, "block": 1.0}))
    if not 0.0 <= bands.review <= bands.block <= 1.0:
        raise ValueError(f"bands must satisfy 0 <= review <= block <= 1, got {bands}")
    # Refuse to serve an artifact that does not reproduce its frozen probabilities (G8).
    if raw.get("require_parity", True):
        parity = verify(model, model_path)
        parity_rows = parity.n_rows
    else:
        parity_rows = 0 if not all(p.exists() for p in frozen_paths(model_path)) else -1
    return ServingState(
        model=model,
        threshold=threshold,
        bands=bands,
        model_version=f"{raw['model_version']}@{digest}",
        parity_rows=parity_rows,
        info=dict(raw.get("model_info", {})),
    )


def classify(p: float, bands: Bands) -> tuple[RiskLevel, Action]:
    if p >= bands.block:
        return "high", "block"
    if p >= bands.review:
        return "medium", "review"
    return "low", "approve"


def _response(state: ServingState, transaction_id: int, p: float) -> PredictionResponse:
    level, action = classify(p, state.bands)
    return PredictionResponse(
        transaction_id=transaction_id,
        fraud_probability=p,
        decision="decline" if p >= state.threshold else "approve",
        risk_level=level,
        action=action,
        threshold=state.threshold,
        model_version=state.model_version,
    )


def score(state: ServingState, payload: dict[str, Any]) -> PredictionResponse:
    p = float(state.model.predict_proba(request_to_frame(payload))[0, 1])
    return _response(state, int(payload[schema.ID_COL]), p)


def score_batch(state: ServingState, payloads: list[dict[str, Any]]) -> BatchPredictionResponse:
    """Score many rows in one pass and return them ranked, highest risk first."""
    probabilities = np.asarray(state.model.predict_proba(payloads_to_frame(payloads))[:, 1])
    order = np.argsort(-probabilities, kind="stable")
    ranked = [
        RankedPrediction(
            rank=rank,
            **_response(
                state, int(payloads[i][schema.ID_COL]), float(probabilities[i])
            ).model_dump(),
        )
        for rank, i in enumerate(order.tolist(), start=1)
    ]
    counts: dict[Action, int] = {"approve": 0, "review": 0, "block": 0}
    for r in ranked:
        counts[r.action] += 1
    return BatchPredictionResponse(n=len(ranked), ranked=ranked, counts=counts)


def model_info(state: ServingState) -> ModelInfoResponse:
    pipe = state.model.pipeline
    n_inputs = sum(len(cols) for _, _, cols in pipe.named_steps["features"].transformers)
    info = state.info
    return ModelInfoResponse(
        model=str(info.get("model", "xgboost")),
        experiment=str(info.get("experiment", "")),
        version=state.model_version,
        feature_set=str(info.get("feature_set", "")),
        n_inputs=n_inputs,
        primary_metric=str(info.get("primary_metric", "pr_auc")),
        validation_pr_auc=float(info.get("validation_pr_auc", float("nan"))),
        test_pr_auc=float(info.get("test_pr_auc", float("nan"))),
        calibration=str(info.get("calibration", state.model.method)),
        threshold=state.threshold,
        bands=state.bands,
        training_window_days=tuple(info.get("training_window_days", (0, 0))),
        validation_window_days=tuple(info.get("validation_window_days", (0, 0))),
        parity_rows=state.parity_rows,
    )


def create_app(config_path: Path = DEFAULT_CONFIG) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.serving = load_state(config_path)
        yield

    app = FastAPI(title="fraud-risk-scoring", version="0.3.0", lifespan=lifespan)

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

    @app.get("/model-info", response_model=ModelInfoResponse)
    def info(request: Request) -> ModelInfoResponse:
        return model_info(request.app.state.serving)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: TransactionRequest, request: Request) -> PredictionResponse:  # type: ignore[valid-type]
        return score(request.app.state.serving, body.model_dump())  # type: ignore[attr-defined]

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(body: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
        payloads = [t.model_dump() for t in body.transactions]  # type: ignore[attr-defined]
        return score_batch(request.app.state.serving, payloads)

    return app


app = create_app()
