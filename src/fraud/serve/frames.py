"""Build the one-row frame the pipeline expects from a validated request payload.

Shared by the API and the frozen-sample parity check, so both score exactly the
same object the training pipeline was fit on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraud.data import schema
from fraud.serve.schemas import IDENTITY_FIELDS, MAX_INT, MAX_TEXT, REQUIRED

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
    """Many validated requests -> one frame, column by column.

    Equal to concatenating :func:`request_to_frame` over the payloads (a test
    asserts it, dtypes included) — that construction is one Series per field per
    row and took 19 s for a 1,000-row batch; this is one Series per field.
    """
    columns: dict[str, pd.Series] = {}
    for col in INPUT_COLUMNS:
        values: list[Any] = [p.get(col) for p in payloads]
        if col in (schema.ID_COL, schema.TIME_COL):
            columns[col] = pd.Series([int(v) for v in values], dtype="int64")
        elif col in _STR_COLS:
            columns[col] = pd.Series([None if v is None else str(v) for v in values], dtype="str")
        else:
            columns[col] = pd.Series(
                [float("nan") if v is None else float(v) for v in values], dtype="float64"
            )
    columns[schema.HAS_IDENTITY_COL] = pd.Series(
        [any(p.get(c) is not None for c in IDENTITY_FIELDS) for p in payloads], dtype="bool"
    )
    return pd.DataFrame(columns)


class TableError(ValueError):
    """An upload that breaks the input contract, with every offending cell located.

    ``errors`` has FastAPI's validation shape, so a CSV and a JSON batch are refused
    the same way: ``{"loc": ["file", row, column], "msg": ..., "type": ...}``, with
    ``row`` the 1-based data row (the header is not counted).
    """

    MAX_REPORTED = 50

    def __init__(self, errors: list[dict[str, Any]], total: int) -> None:
        self.errors = errors
        self.total = total
        more = f" (first {len(errors)} shown)" if total > len(errors) else ""
        super().__init__(f"{total} invalid value(s){more}: {errors[0]['msg']}")


_INTEGER = r"^\s*\d+(\.0*)?\s*$"  # 12, 12.0 — never 1.9, -1, 1e3 or a blank


def table_to_frame(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """A whole uploaded table (e.g. a CSV read as strings) -> the frame the pipeline expects.

    Vectorised twin of :func:`request_to_frame`, under the same contract as the JSON
    request model (`fraud.serve.schemas`): ids and times are exact non-negative integers
    within int64, numbers are finite, the amount is positive, required fields are not
    blank, text is at most ``MAX_TEXT`` characters, ids are unique, and a blank cell is a
    missing value. Unknown columns are ignored and reported back. A missing required
    column raises ``ValueError``; bad cells raise :class:`TableError` naming each one.
    """
    missing = [c for c in REQUIRED if c not in table.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")
    ignored = sorted(c for c in table.columns if c not in INPUT_COLUMNS)
    errors: list[dict[str, Any]] = []
    total = 0

    def flag(bad: np.ndarray, col: str, msg: str, kind: str) -> None:
        nonlocal total
        rows = np.flatnonzero(bad)
        total += len(rows)
        for r in rows[: max(TableError.MAX_REPORTED - len(errors), 0)]:
            errors.append({"loc": ["file", int(r) + 1, col], "msg": msg, "type": kind})

    columns: dict[str, pd.Series] = {}
    for col in INPUT_COLUMNS:
        raw = table[col] if col in table.columns else pd.Series([None] * len(table))
        raw = raw.where(raw.notna() & (raw.astype("str").str.strip() != ""), other=None)
        blank = raw.isna().to_numpy()
        if col in REQUIRED:
            flag(blank, col, "Field required", "missing")
        if col in _STR_COLS:
            too_long = (raw.astype("str").str.len() > MAX_TEXT).to_numpy() & ~blank
            flag(
                too_long,
                col,
                f"String should have at most {MAX_TEXT} characters",
                "string_too_long",
            )
            columns[col] = pd.Series(raw.to_numpy(), dtype="str")
            continue
        if col in (schema.ID_COL, schema.TIME_COL):
            text = raw.astype("str")
            exact = text.str.match(_INTEGER).to_numpy() & ~blank
            flag(~exact & ~blank, col, "Input should be a non-negative whole number", "int_parsing")
            digits = text.where(exact, "0").str.strip().str.split(".").str[0].str.lstrip("0")
            digits = digits.where(digits != "", "0")
            big = digits.str.len().to_numpy() > len(str(MAX_INT))
            big |= ~big & (digits.str.zfill(len(str(MAX_INT))) > str(MAX_INT)).to_numpy()
            flag(big, col, f"Input should be less than or equal to {MAX_INT}", "less_than_equal")
            columns[col] = digits.where(~big, "0").astype("int64")
            continue
        numeric = pd.to_numeric(raw, errors="coerce").astype("float64")
        values = numeric.to_numpy()
        flag(np.isnan(values) & ~blank, col, "Input should be a valid number", "float_parsing")
        flag(np.isinf(values), col, "Input should be a finite number", "finite_number")
        if col == "TransactionAmt":
            flag(values <= 0, col, "Input should be greater than 0", "greater_than")
        columns[col] = numeric
    ids = columns[schema.ID_COL]
    flag(ids.duplicated(keep=False).to_numpy(), schema.ID_COL,
         f"{schema.ID_COL} must be unique within an upload", "value_error")  # fmt: skip
    if total:
        raise TableError(errors, total)
    frame = pd.DataFrame(columns)
    frame[schema.HAS_IDENTITY_COL] = frame[list(IDENTITY_FIELDS)].notna().any(axis=1)
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
