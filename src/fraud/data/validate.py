"""Data-contract checks for the raw IEEE-CIS tables (G9: assumptions become tests).

Each check raises :class:`SchemaError` with a message that names the column and
the violation. ``fraud.data.load`` calls these on every read;
``scripts/validate_data.py`` runs them on their own and prints a report.
"""

from __future__ import annotations

import pandas as pd

from fraud.data import schema

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

# Row counts as published on the competition data page; used to detect a
# truncated or altered download.
EXPECTED_ROWS: dict[str, int] = {TRANSACTION_FILE: 590_540, IDENTITY_FILE: 144_233}


class SchemaError(ValueError):
    """Raised when a raw table violates the data contract."""


def check_columns(df: pd.DataFrame, expected: tuple[str, ...], name: str) -> None:
    actual = tuple(df.columns)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise SchemaError(f"{name}: columns differ; missing={missing} extra={extra}")


def check_dtypes(df: pd.DataFrame, str_cols: frozenset[str], name: str) -> None:
    bad: list[str] = []
    for col in df.columns:
        dt = df[col].dtype
        expect_str = col in str_cols
        is_str = pd.api.types.is_string_dtype(dt)
        is_num = pd.api.types.is_numeric_dtype(dt)
        if (expect_str and not is_str) or (not expect_str and not is_num):
            bad.append(f"{col}={dt}")
    if bad:
        raise SchemaError(f"{name}: unexpected dtypes: {bad[:10]}")


def check_non_null(df: pd.DataFrame, cols: frozenset[str], name: str) -> None:
    nulls = {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}
    if nulls:
        raise SchemaError(f"{name}: nulls in columns expected non-null: {nulls}")


def check_unique_id(df: pd.DataFrame, name: str) -> None:
    if not df[schema.ID_COL].is_unique:
        raise SchemaError(f"{name}: {schema.ID_COL} is not unique")


def check_row_count(df: pd.DataFrame, name: str) -> None:
    """Raise if the frame does not have the published row count for ``name``."""
    expected = EXPECTED_ROWS[name]
    if len(df) != expected:
        raise SchemaError(f"{name}: {len(df):,} rows, expected {expected:,}")


def validate_transactions(df: pd.DataFrame) -> None:
    name = TRANSACTION_FILE
    check_columns(df, schema.TRANSACTION_COLS, name)
    check_dtypes(df, schema.TRANSACTION_STR_COLS, name)
    check_non_null(df, schema.TRANSACTION_NON_NULL, name)
    check_unique_id(df, name)
    labels = set(df[schema.TARGET_COL].unique())
    if not labels <= {0, 1}:
        raise SchemaError(f"{name}: {schema.TARGET_COL} has values outside {{0, 1}}: {labels}")
    rate = float(df[schema.TARGET_COL].mean())
    lo, hi = schema.LABEL_RATE_BOUNDS
    if not lo <= rate <= hi:
        raise SchemaError(f"{name}: label rate {rate:.4f} outside [{lo}, {hi}]")
    if (df[schema.TIME_COL] <= 0).any():
        raise SchemaError(f"{name}: {schema.TIME_COL} must be positive")
    if (df["TransactionAmt"] <= 0).any():
        raise SchemaError(f"{name}: TransactionAmt must be positive")


def validate_identity(df: pd.DataFrame) -> None:
    name = IDENTITY_FILE
    check_columns(df, schema.IDENTITY_COLS, name)
    check_dtypes(df, schema.IDENTITY_STR_COLS, name)
    check_non_null(df, schema.IDENTITY_NON_NULL, name)
    check_unique_id(df, name)


def validate_join(tx: pd.DataFrame, idn: pd.DataFrame, merged: pd.DataFrame) -> None:
    """The left join kept every transaction, duplicated none, and flagged identity correctly."""
    if len(merged) != len(tx):
        raise SchemaError(f"join changed row count: {len(tx)} -> {len(merged)}")
    if not merged[schema.ID_COL].is_unique:
        raise SchemaError("join duplicated transactions")
    if int(merged[schema.HAS_IDENTITY_COL].sum()) != len(idn):
        raise SchemaError(
            f"{schema.HAS_IDENTITY_COL} marks {int(merged[schema.HAS_IDENTITY_COL].sum())} rows, "
            f"identity has {len(idn)}"
        )
    coverage = float(merged[schema.HAS_IDENTITY_COL].mean())
    lo, hi = schema.IDENTITY_COVERAGE_BOUNDS
    if not lo <= coverage <= hi:
        raise SchemaError(f"identity coverage {coverage:.3f} outside [{lo}, {hi}]")
