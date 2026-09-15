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


def table_to_frame(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """A whole uploaded table (e.g. a CSV read as strings) -> the frame the pipeline expects.

    Vectorised twin of :func:`request_to_frame`: same columns, same dtypes, same
    ``has_identity`` rule. Unknown columns are ignored and reported back; a
    missing required column or an unparsable number raises ``ValueError``.
    """
    required = (schema.ID_COL, schema.TIME_COL, "TransactionAmt", "ProductCD", "card1")
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")
    ignored = sorted(c for c in table.columns if c not in INPUT_COLUMNS)
    columns: dict[str, pd.Series] = {}
    for col in INPUT_COLUMNS:
        raw = table[col] if col in table.columns else pd.Series([None] * len(table))
        raw = raw.where(raw.notna() & (raw.astype("str").str.strip() != ""), other=None)
        if col in _STR_COLS:
            columns[col] = pd.Series(raw.to_numpy(), dtype="str")
            continue
        try:
            numeric = pd.to_numeric(raw, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"column {col!r} has a non-numeric value: {exc}") from exc
        if col in (schema.ID_COL, schema.TIME_COL):
            if numeric.isna().any():
                raise ValueError(f"column {col!r} has empty values")
            columns[col] = pd.Series(numeric.to_numpy(), dtype="float64").astype("int64")
        else:
            columns[col] = pd.Series(numeric.to_numpy(), dtype="float64")
    frame = pd.DataFrame(columns)
    frame[schema.HAS_IDENTITY_COL] = frame[list(IDENTITY_FIELDS)].notna().any(axis=1)
    if (frame["TransactionAmt"] <= 0).any():
        raise ValueError("TransactionAmt must be positive")
    if not frame[schema.ID_COL].is_unique:
        raise ValueError(f"{schema.ID_COL} must be unique within an upload")
    return frame, ignored


_SKIP = (schema.ID_COL, schema.TARGET_COL)
MONITORED_TX = tuple(c for c in schema.TRANSACTION_COLS if c not in _SKIP)
MONITORED_ID = tuple(c for c in schema.IDENTITY_COLS if c != schema.ID_COL)


def _text(value: Any) -> str | None:
    return None if pd.isna(value) else str(value)


def monitored_fields(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The compact per-row input snapshot the monitor compares against its reference."""
    hour = (frame[schema.TIME_COL] % 86_400) // 3600
    n_tx_missing = frame[list(MONITORED_TX)].isna().sum(axis=1)
    n_id_missing = frame[list(MONITORED_ID)].isna().sum(axis=1)
    out: list[dict[str, Any]] = []
    for i in range(len(frame)):
        row = frame.iloc[i]
        out.append(
            {
                "transaction_id": int(row[schema.ID_COL]),
                "amount": float(row["TransactionAmt"]),
                "product": str(row["ProductCD"]),
                "card4": _text(row["card4"]),
                "card6": _text(row["card6"]),
                "device_type": _text(row["DeviceType"]),
                "has_identity": int(bool(row[schema.HAS_IDENTITY_COL])),
                "addr1_missing": int(pd.isna(row["addr1"])),
                "p_email_present": int(not pd.isna(row["P_emaildomain"])),
                "r_email_present": int(not pd.isna(row["R_emaildomain"])),
                "n_missing_transaction": int(n_tx_missing.iloc[i]),
                "n_missing_identity": int(n_id_missing.iloc[i]),
                "hour": int(hour.iloc[i]),
            }
        )
    return out
