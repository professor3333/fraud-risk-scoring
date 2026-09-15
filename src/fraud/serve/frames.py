"""Build the one-row frame the pipeline expects from a validated request payload.

Shared by the API and the frozen-sample parity check, so both score exactly the
same object the training pipeline was fit on.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fraud.data import schema
from fraud.serve.schemas import IDENTITY_FIELDS

INPUT_COLUMNS: tuple[str, ...] = (
    tuple(c for c in schema.TRANSACTION_COLS if c != schema.TARGET_COL) + IDENTITY_FIELDS
)
_STR_COLS = schema.TRANSACTION_STR_COLS | schema.IDENTITY_STR_COLS


def _typed_column(col: str, value: Any) -> pd.Series:
    if col in (schema.ID_COL, schema.TIME_COL):
        return pd.Series([int(value)], dtype="int64")
    if col in _STR_COLS:
        return pd.Series([None if value is None else str(value)], dtype="str")
    return pd.Series([float("nan") if value is None else float(value)], dtype="float64")


def request_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """One validated request -> a one-row frame shaped like the training frame."""
    columns = {c: _typed_column(c, payload.get(c)) for c in INPUT_COLUMNS}
    columns[schema.HAS_IDENTITY_COL] = pd.Series(
        [any(payload.get(c) is not None for c in IDENTITY_FIELDS)], dtype="bool"
    )
    return pd.DataFrame(columns)


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    """The inverse for stored samples: a joined-frame row -> a request payload."""
    out: dict[str, Any] = {}
    for col, val in row.items():
        if col not in INPUT_COLUMNS or pd.isna(val):
            continue
        if col in (schema.ID_COL, schema.TIME_COL):
            out[col] = int(val)
        elif col in _STR_COLS:
            out[col] = str(val)
        else:
            out[col] = float(val)
    return out


def payloads_to_frame(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.concat([request_to_frame(p) for p in payloads], ignore_index=True)
