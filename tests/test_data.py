"""Data tests: the raw files honour the contract in fraud.data.schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud.data import schema
from fraud.data.load import (
    SchemaError,
    join_transaction_identity,
    load_train,
    validate_identity,
    validate_transactions,
)


def test_transaction_columns_and_shape(tx: pd.DataFrame) -> None:
    assert tuple(tx.columns) == schema.TRANSACTION_COLS
    assert tx.shape[1] == 394


def test_identity_columns_and_shape(idn: pd.DataFrame) -> None:
    assert tuple(idn.columns) == schema.IDENTITY_COLS
    assert idn.shape[1] == 41


def test_transaction_ids_unique(tx: pd.DataFrame, idn: pd.DataFrame) -> None:
    assert tx[schema.ID_COL].is_unique
    assert idn[schema.ID_COL].is_unique


def test_string_columns_have_string_dtype(tx: pd.DataFrame, idn: pd.DataFrame) -> None:
    for col in schema.TRANSACTION_STR_COLS:
        assert pd.api.types.is_string_dtype(tx[col]), col
    for col in schema.IDENTITY_STR_COLS:
        assert pd.api.types.is_string_dtype(idn[col]), col


def test_label_is_binary_and_rare(tx: pd.DataFrame) -> None:
    assert set(tx[schema.TARGET_COL].unique()) <= {0, 1}
    lo, hi = schema.LABEL_RATE_BOUNDS
    assert lo <= tx[schema.TARGET_COL].mean() <= hi


def test_join_keeps_every_transaction_and_flags_identity(
    tx: pd.DataFrame, idn: pd.DataFrame
) -> None:
    merged = join_transaction_identity(tx, idn)
    assert len(merged) == len(tx)
    assert merged[schema.ID_COL].is_unique
    assert merged[schema.HAS_IDENTITY_COL].dtype == bool
    assert merged[schema.HAS_IDENTITY_COL].sum() == len(idn)
    # Identity is missing for most rows: absence is a signal, not a bug.
    assert merged[schema.HAS_IDENTITY_COL].mean() < 0.5
    # Rows without identity have all identity columns null; rows with it have a device type.
    without = merged.loc[~merged[schema.HAS_IDENTITY_COL], "DeviceType"]
    with_ = merged.loc[merged[schema.HAS_IDENTITY_COL], "DeviceType"]
    assert without.isna().all()
    assert with_.notna().all()


def test_join_rejects_orphan_identity_rows(tx: pd.DataFrame, idn: pd.DataFrame) -> None:
    bad = idn.copy()
    bad.loc[bad.index[0], schema.ID_COL] = -1
    with pytest.raises(SchemaError, match="no matching transaction"):
        join_transaction_identity(tx, bad)


def test_validate_rejects_missing_column(tx: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="missing=\\['V1'\\]"):
        validate_transactions(tx.drop(columns=["V1"]))


def test_validate_rejects_duplicate_id(idn: pd.DataFrame) -> None:
    dup = pd.concat([idn, idn.iloc[:1]], ignore_index=True)
    with pytest.raises(SchemaError, match="not unique"):
        validate_identity(dup)


def test_validate_rejects_null_in_required_column(tx: pd.DataFrame) -> None:
    bad = tx.copy()
    bad.loc[bad.index[0], "TransactionAmt"] = float("nan")
    with pytest.raises(SchemaError, match="expected non-null"):
        validate_transactions(bad)


def test_validate_rejects_label_outside_binary(tx: pd.DataFrame) -> None:
    bad = tx.copy()
    bad.loc[bad.index[0], schema.TARGET_COL] = 2
    with pytest.raises(SchemaError, match="outside"):
        validate_transactions(bad)


def test_load_train_cache_roundtrip(fixture_raw_dir: Path, tmp_path: Path) -> None:
    first = load_train(fixture_raw_dir, cache_dir=tmp_path)
    assert (tmp_path / "train.parquet").exists()
    second = load_train(fixture_raw_dir, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(first, second)


# --- full dataset ---------------------------------------------------------------


@pytest.mark.slow
def test_full_dataset_honours_contract(full_raw_dir: Path) -> None:
    merged = load_train(full_raw_dir)
    assert len(merged) == 590_540
    assert merged[schema.HAS_IDENTITY_COL].sum() == 144_233
    assert 0.03 <= merged[schema.TARGET_COL].mean() <= 0.04
    assert merged[schema.TIME_COL].is_monotonic_increasing
