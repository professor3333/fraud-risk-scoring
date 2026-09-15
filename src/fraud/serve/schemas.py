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
    decision: Literal["approve", "decline"]
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    threshold: float
    parity_rows: int  # frozen rows re-scored at startup; the service refuses to start on a mismatch
