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
    """The authoritative output: one probability, one action, one risk level.

    The evaluation threshold (0.08, ADR 0006) is reporting material only; the
    three-action review policy is the production policy.
    """

    transaction_id: int
    fraud_probability: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel  # low / medium / high
    action: Action  # approve / review / block — the one command for the payment system
    model_version: str


Policy = Literal["rank", "threshold"]


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[TransactionRequest] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]
    # Analyst capacity for this batch. With the default "rank" policy: block by
    # threshold, review the review_budget highest remaining scores, approve the
    # rest (docs/review_policy.md). None = the server's default budget.
    review_budget: int | None = Field(default=None, ge=0)
    policy: Policy = "rank"  # "threshold" = fixed review threshold instead of a budget


class RankedPrediction(PredictionResponse):
    rank: int = Field(ge=1)


class PolicyApplied(BaseModel):
    policy: Policy
    block_threshold: float
    review_budget: int | None  # the budget used (rank policy)
    review_threshold: float | None  # the fixed threshold (threshold policy)
    review_cutoff: float | None  # lowest probability actually reviewed in this batch


class BatchPredictionResponse(BaseModel):
    n: int
    ranked: list[RankedPrediction]  # highest fraud probability first
    counts: dict[Action, int]
    policy: PolicyApplied


class Bands(BaseModel):
    review: float
    block: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    parity_rows: int  # frozen rows re-scored at startup; the service refuses to start on a mismatch


class ModelInfoResponse(BaseModel):
    model: str
    default_policy: Policy
    default_review_budget: int
    experiment: str
    version: str
    feature_set: str
    n_inputs: int
    primary_metric: str
    validation_pr_auc: float
    test_pr_auc: float
    calibration: str
    bands: Bands  # production policy: block at .block; review by budget (rank) or down to .review
    training_window_days: tuple[int, int]
    validation_window_days: tuple[int, int]
    parity_rows: int


class ScoredRow(BaseModel):
    rank: int
    transaction_id: int
    fraud_probability: float
    risk_level: RiskLevel
    action: Action
    amount: float
    product: str
    card_type: str | None
    has_identity: bool
    details: dict[str, str | float | int | bool]  # the non-empty fields, for inspection


class CsvSummary(BaseModel):
    analysed: int
    flagged: int  # review + block
    high_risk: int  # block
    review: int
    approve: int
    average_fraud_probability: float
    ignored_columns: list[str]
    model_version: str
    policy: PolicyApplied


class CsvPredictionResponse(BaseModel):
    summary: CsvSummary
    rows: list[ScoredRow]  # ranked, highest risk first
