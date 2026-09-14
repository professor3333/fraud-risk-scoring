"""Load, validate, and join the raw IEEE-CIS training tables.

Functions here read files and check the data contract in :mod:`fraud.data.schema`.
They never fit anything and never drop or impute values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud.data import schema

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
CACHE_FILE = "train.parquet"


class SchemaError(ValueError):
    """Raised when a raw table violates the data contract."""


def _read_csv(path: Path, str_cols: frozenset[str]) -> pd.DataFrame:
    dtype = dict.fromkeys(str_cols, "str")
    return pd.read_csv(path, dtype=dtype)


def load_transactions(raw_dir: Path) -> pd.DataFrame:
    df = _read_csv(raw_dir / TRANSACTION_FILE, schema.TRANSACTION_STR_COLS)
    validate_transactions(df)
    return df


def load_identity(raw_dir: Path) -> pd.DataFrame:
    df = _read_csv(raw_dir / IDENTITY_FILE, schema.IDENTITY_STR_COLS)
    validate_identity(df)
    return df


def _check_columns(df: pd.DataFrame, expected: tuple[str, ...], name: str) -> None:
    actual = tuple(df.columns)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise SchemaError(f"{name}: columns differ; missing={missing} extra={extra}")


def _check_dtypes(df: pd.DataFrame, str_cols: frozenset[str], name: str) -> None:
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


def _check_non_null(df: pd.DataFrame, cols: frozenset[str], name: str) -> None:
    nulls = {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}
    if nulls:
        raise SchemaError(f"{name}: nulls in columns expected non-null: {nulls}")


def _check_unique_id(df: pd.DataFrame, name: str) -> None:
    if not df[schema.ID_COL].is_unique:
        raise SchemaError(f"{name}: {schema.ID_COL} is not unique")


def validate_transactions(df: pd.DataFrame) -> None:
    name = TRANSACTION_FILE
    _check_columns(df, schema.TRANSACTION_COLS, name)
    _check_dtypes(df, schema.TRANSACTION_STR_COLS, name)
    _check_non_null(df, schema.TRANSACTION_NON_NULL, name)
    _check_unique_id(df, name)
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
    _check_columns(df, schema.IDENTITY_COLS, name)
    _check_dtypes(df, schema.IDENTITY_STR_COLS, name)
    _check_non_null(df, schema.IDENTITY_NON_NULL, name)
    _check_unique_id(df, name)


def join_transaction_identity(tx: pd.DataFrame, idn: pd.DataFrame) -> pd.DataFrame:
    """Left-join identity onto transactions, keeping every transaction.

    Adds a boolean ``has_identity`` column: identity absence is a signal and must
    not be confused with a missing value inside an identity column.
    """
    orphan = ~idn[schema.ID_COL].isin(tx[schema.ID_COL])
    if orphan.any():
        raise SchemaError(f"{int(orphan.sum())} identity rows have no matching transaction")
    merged = tx.merge(idn, on=schema.ID_COL, how="left", validate="one_to_one")
    if len(merged) != len(tx):
        raise SchemaError(f"join changed row count: {len(tx)} -> {len(merged)}")
    flag = merged[schema.ID_COL].isin(idn[schema.ID_COL]).rename(schema.HAS_IDENTITY_COL)
    merged = pd.concat([merged, flag], axis=1)
    coverage = float(merged[schema.HAS_IDENTITY_COL].mean())
    lo, hi = schema.IDENTITY_COVERAGE_BOUNDS
    if not lo <= coverage <= hi:
        raise SchemaError(f"identity coverage {coverage:.3f} outside [{lo}, {hi}]")
    return merged


def load_train(raw_dir: Path, cache_dir: Path | None = None) -> pd.DataFrame:
    """Return the joined, validated training frame, using a Parquet cache if given.

    The cache is a pure convenience: it holds exactly what the CSV path produces.
    """
    if cache_dir is not None:
        cache = cache_dir / CACHE_FILE
        if cache.exists():
            return pd.read_parquet(cache)
    df = join_transaction_identity(load_transactions(raw_dir), load_identity(raw_dir))
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_dir / CACHE_FILE, index=False)
    return df
