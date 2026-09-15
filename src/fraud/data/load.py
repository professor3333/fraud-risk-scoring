"""Load and join the raw IEEE-CIS training tables.

Reads files, applies the checks in :mod:`fraud.data.validate`, and left-joins
identity onto transactions. Nothing here is fit; nothing is dropped or imputed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud.data import schema
from fraud.data.validate import (
    IDENTITY_FILE,
    TRANSACTION_FILE,
    SchemaError,
    validate_identity,
    validate_join,
    validate_transactions,
)

__all__ = [
    "CACHE_FILE",
    "IDENTITY_FILE",
    "TRANSACTION_FILE",
    "SchemaError",
    "join_transaction_identity",
    "load_identity",
    "load_train",
    "load_transactions",
    "validate_identity",
    "validate_transactions",
]

CACHE_FILE = "train.parquet"


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


def join_transaction_identity(tx: pd.DataFrame, idn: pd.DataFrame) -> pd.DataFrame:
    """Left-join identity onto transactions, keeping every transaction.

    Adds a boolean ``has_identity`` column: identity absence is a signal and must
    not be confused with a missing value inside an identity column.
    """
    orphan = ~idn[schema.ID_COL].isin(tx[schema.ID_COL])
    if orphan.any():
        raise SchemaError(f"{int(orphan.sum())} identity rows have no matching transaction")
    merged = tx.merge(idn, on=schema.ID_COL, how="left", validate="one_to_one")
    flag = merged[schema.ID_COL].isin(idn[schema.ID_COL]).rename(schema.HAS_IDENTITY_COL)
    merged = pd.concat([merged, flag], axis=1)
    validate_join(tx, idn, merged)
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
