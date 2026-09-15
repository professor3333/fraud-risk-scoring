"""FastAPI inference service: the calibrated pipeline, loaded once, scoring raw rows (G8)."""

from __future__ import annotations

import hashlib
import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from fraud.data import schema
from fraud.evaluate.policy import apply_rank_policy
from fraud.evaluate.threshold import load_threshold_config
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.frames import payloads_to_frame, request_to_frame, table_to_frame
from fraud.serve.parity import frozen_paths, verify
from fraud.serve.schemas import (
    Action,
    Bands,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CsvPredictionResponse,
    CsvSummary,
    HealthResponse,
    ModelInfoResponse,
    Policy,
    PolicyApplied,
    PredictionResponse,
    RankedPrediction,
    RiskLevel,
    ScoredRow,
    TransactionRequest,
)

MAX_CSV_ROWS = 5_000

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
    default_review_budget: int


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
        default_review_budget=int(raw.get("default_review_budget", 200)),
    )


def classify(p: float, bands: Bands) -> tuple[RiskLevel, Action]:
    if p >= bands.block:
        return "high", "block"
    if p >= bands.review:
        return "medium", "review"
    return "low", "approve"


LEVEL_FOR_ACTION: dict[Action, RiskLevel] = {"block": "high", "review": "medium", "approve": "low"}


def _response(
    state: ServingState, transaction_id: int, p: float, action: Action | None = None
) -> PredictionResponse:
    if action is None:
        level, action = classify(p, state.bands)
    else:
        level = LEVEL_FOR_ACTION[action]
    return PredictionResponse(
        transaction_id=transaction_id,
        fraud_probability=p,
        decision="decline" if p >= state.threshold else "approve",
        risk_level=level,
        action=action,
        threshold=state.threshold,
        model_version=state.model_version,
    )


def assign_actions(
    state: ServingState, probabilities: np.ndarray, policy: Policy, review_budget: int | None
) -> tuple[list[Action], PolicyApplied]:
    """Actions for a scored batch: rank-based (default) or fixed-threshold bands."""
    if policy == "rank":
        budget = state.default_review_budget if review_budget is None else review_budget
        actions, cutoff = apply_rank_policy(probabilities, state.bands.block, budget)
        applied = PolicyApplied(
            policy="rank",
            block_threshold=state.bands.block,
            review_budget=budget,
            review_threshold=None,
            review_cutoff=cutoff,
        )
        return [str(a) for a in actions], applied  # type: ignore[misc]
    actions_t: list[Action] = [classify(float(p), state.bands)[1] for p in probabilities]
    reviewed = [float(p) for p, a in zip(probabilities, actions_t, strict=True) if a == "review"]
    applied = PolicyApplied(
        policy="threshold",
        block_threshold=state.bands.block,
        review_budget=None,
        review_threshold=state.bands.review,
        review_cutoff=min(reviewed) if reviewed else None,
    )
    return actions_t, applied


def score(state: ServingState, payload: dict[str, Any]) -> PredictionResponse:
    p = float(state.model.predict_proba(request_to_frame(payload))[0, 1])
    return _response(state, int(payload[schema.ID_COL]), p)


def score_batch(
    state: ServingState,
    payloads: list[dict[str, Any]],
    policy: Policy = "rank",
    review_budget: int | None = None,
) -> BatchPredictionResponse:
    """Score many rows in one pass, apply the policy, return them ranked, highest risk first."""
    probabilities = np.asarray(state.model.predict_proba(payloads_to_frame(payloads))[:, 1])
    actions, applied = assign_actions(state, probabilities, policy, review_budget)
    order = np.argsort(-probabilities, kind="stable")
    ranked = []
    for rank, i in enumerate(order.tolist(), start=1):
        single = _response(
            state, int(payloads[i][schema.ID_COL]), float(probabilities[i]), actions[i]
        )
        ranked.append(RankedPrediction(rank=rank, **single.model_dump()))
    counts: dict[Action, int] = {"approve": 0, "review": 0, "block": 0}
    for r in ranked:
        counts[r.action] += 1
    return BatchPredictionResponse(n=len(ranked), ranked=ranked, counts=counts, policy=applied)


def score_table(
    state: ServingState,
    table: pd.DataFrame,
    policy: Policy = "rank",
    review_budget: int | None = None,
) -> CsvPredictionResponse:
    """Score an uploaded table; apply the policy; rank rows and summarise for the analyst view."""
    frame, ignored = table_to_frame(table)
    probabilities = np.asarray(state.model.predict_proba(frame)[:, 1], dtype=float)
    actions, applied = assign_actions(state, probabilities, policy, review_budget)
    order = np.argsort(-probabilities, kind="stable")
    rows: list[ScoredRow] = []
    counts = {"approve": 0, "review": 0, "block": 0}
    for rank, i in enumerate(order.tolist(), start=1):
        p = float(probabilities[i])
        action = actions[i]
        level = LEVEL_FOR_ACTION[action]
        counts[action] += 1
        record = frame.iloc[i]
        details = {
            str(k): (v.item() if hasattr(v, "item") else v)
            for k, v in record.items()
            if not (isinstance(v, float) and np.isnan(v)) and v is not None and v == v
        }
        card_type = record["card6"]
        rows.append(
            ScoredRow(
                rank=rank,
                transaction_id=int(record[schema.ID_COL]),
                fraud_probability=p,
                risk_level=level,
                action=action,
                amount=float(record["TransactionAmt"]),
                product=str(record["ProductCD"]),
                card_type=None if pd.isna(card_type) else str(card_type),
                has_identity=bool(record[schema.HAS_IDENTITY_COL]),
                details=details,
            )
        )
    summary = CsvSummary(
        analysed=len(rows),
        flagged=counts["review"] + counts["block"],
        high_risk=counts["block"],
        review=counts["review"],
        approve=counts["approve"],
        average_fraud_probability=float(probabilities.mean()) if len(rows) else 0.0,
        ignored_columns=ignored,
        model_version=state.model_version,
        threshold=state.threshold,
        bands=state.bands,
        policy=applied,
    )
    return CsvPredictionResponse(summary=summary, rows=rows)


def model_info(state: ServingState) -> ModelInfoResponse:
    pipe = state.model.pipeline
    n_inputs = sum(len(cols) for _, _, cols in pipe.named_steps["features"].transformers)
    info = state.info
    return ModelInfoResponse(
        model=str(info.get("model", "xgboost")),
        default_policy="rank",
        default_review_budget=state.default_review_budget,
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
        return (STATIC / "dashboard.html").read_text()

    @app.get("/single", response_class=HTMLResponse, include_in_schema=False)
    def single() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/sample.csv", include_in_schema=False)
    def sample_csv() -> FileResponse:
        return FileResponse(STATIC / "sample_transactions.csv", media_type="text/csv")

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
        return score_batch(request.app.state.serving, payloads, body.policy, body.review_budget)

    @app.post("/predict/csv", response_model=CsvPredictionResponse)
    async def predict_csv(
        request: Request,
        file: UploadFile,
        review_budget: int | None = None,
        policy: Policy = "rank",
    ) -> CsvPredictionResponse:
        raw = await file.read()
        try:
            table = pd.read_csv(io.BytesIO(raw), dtype="str", keep_default_na=False)
        except (ValueError, pd.errors.ParserError) as exc:
            raise HTTPException(422, f"could not parse CSV: {exc}") from exc
        if len(table) == 0:
            raise HTTPException(422, "the CSV has no rows")
        if len(table) > MAX_CSV_ROWS:
            raise HTTPException(422, f"at most {MAX_CSV_ROWS} rows per upload (got {len(table)})")
        try:
            return score_table(request.app.state.serving, table, policy, review_budget)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
