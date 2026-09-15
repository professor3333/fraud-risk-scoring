"""Request / response contracts for the API, generated from the raw-data schema.

A request carries one transaction with the provider's column names. Identity
fields are optional: leaving all of them out means "no identity record", which
becomes ``has_identity = False`` — the same signal the training data had.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud.data import schema

_TX_INPUT_COLS = tuple(c for c in schema.TRANSACTION_COLS if c != schema.TARGET_COL)
_ID_INPUT_COLS = tuple(c for c in schema.IDENTITY_COLS if c != schema.ID_COL)
IDENTITY_FIELDS: tuple[str, ...] = _ID_INPUT_COLS

RiskLevel = Literal["low", "medium", "high"]
Action = Literal["approve", "review", "block"]


def _field_type(col: str, str_cols: frozenset[str], required: bool) -> tuple[Any, Any]:
    if col in str_cols:
        return (str, ...) if required else (str | None, None)
    return (float, ...) if required else (float | None, None)


_REQUIRED = {schema.ID_COL, schema.TIME_COL, "TransactionAmt", "ProductCD", "card1"}

_fields: dict[str, Any] = {}
for _col in _TX_INPUT_COLS:
    if _col == schema.ID_COL or _col == schema.TIME_COL:
        _fields[_col] = (int, Field(..., ge=0))
    elif _col == "TransactionAmt":
        _fields[_col] = (float, Field(..., gt=0))
    else:
        _fields[_col] = _field_type(_col, schema.TRANSACTION_STR_COLS, _col in _REQUIRED)
for _col in _ID_INPUT_COLS:
    _fields[_col] = _field_type(_col, schema.IDENTITY_STR_COLS, required=False)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


TransactionRequest = create_model("TransactionRequest", __base__=_Base, **_fields)


class PredictionResponse(BaseModel):
    transaction_id: int
    fraud_probability: float = Field(ge=0.0, le=1.0)
    decision: Literal["approve", "decline"]  # single-threshold decision (ADR 0006)
    risk_level: RiskLevel  # low / medium / high from the review-policy bands
    action: Action  # approve / review / block
    threshold: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[TransactionRequest] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]


class RankedPrediction(PredictionResponse):
    rank: int = Field(ge=1)


class BatchPredictionResponse(BaseModel):
    n: int
    ranked: list[RankedPrediction]  # highest fraud probability first
    counts: dict[Action, int]


class Bands(BaseModel):
    review: float
    block: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    threshold: float
    parity_rows: int  # frozen rows re-scored at startup; the service refuses to start on a mismatch


class ModelInfoResponse(BaseModel):
    model: str
    experiment: str
    version: str
    feature_set: str
    n_inputs: int
    primary_metric: str
    validation_pr_auc: float
    test_pr_auc: float
    calibration: str
    threshold: float
    bands: Bands
    training_window_days: tuple[int, int]
    validation_window_days: tuple[int, int]
    parity_rows: int
