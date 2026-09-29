"""Load and join the raw IEEE-CIS training tables.

Reads files, applies the checks in :mod:`fraud.data.validate`, and left-joins
identity onto transactions. Nothing here is fit; nothing is dropped or imputed.
"""

from __future__ import annotations

import hashlib
import json
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
#: Bump when what load_train produces from the same CSVs changes (dtypes, the join,
#: added columns): a cache built by an older version is then rebuilt, not trusted.
CACHE_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_identity(raw_dir: Path) -> dict[str, object]:
    """What a cache must have been built from to stand in for these raw files."""
    return {
        "cache_version": CACHE_VERSION,
        "sources": {n: _file_sha256(raw_dir / n) for n in (TRANSACTION_FILE, IDENTITY_FILE)},
    }


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

    The cache is a pure convenience: it holds exactly what the CSV path produces from
    the same files. So it is used only when its sidecar records the sha256 of both raw
    files and the current ``CACHE_VERSION``; anything else rebuilds it. Existence alone
    used to be enough, and an edited CSV kept loading its old values from the cache.
    Hashing the ~700 MB of raw CSVs costs about 0.6 s, a quarter of the Parquet read.
    """
    identity = cache_identity(raw_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache, meta = cache_dir / CACHE_FILE, cache_dir / f"{CACHE_FILE}.meta.json"
        if cache.exists() and meta.exists() and json.loads(meta.read_text()) == identity:
            return pd.read_parquet(cache)
    df = join_transaction_identity(load_transactions(raw_dir), load_identity(raw_dir))
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta.unlink(missing_ok=True)  # never a new cache under an old identity
        df.to_parquet(cache, index=False)
        meta.write_text(json.dumps(identity, indent=2))
    return df
