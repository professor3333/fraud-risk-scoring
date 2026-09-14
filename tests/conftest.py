from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud.data.load import load_identity, load_transactions

FIXTURE_RAW = Path(__file__).resolve().parent / "fixtures" / "raw"
PROJECT_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="session")
def fixture_raw_dir() -> Path:
    return FIXTURE_RAW


@pytest.fixture(scope="session")
def tx(fixture_raw_dir: Path) -> pd.DataFrame:
    return load_transactions(fixture_raw_dir)


@pytest.fixture(scope="session")
def idn(fixture_raw_dir: Path) -> pd.DataFrame:
    return load_identity(fixture_raw_dir)


@pytest.fixture(scope="session")
def full_raw_dir() -> Path:
    """The real dataset; only for tests marked ``slow``."""
    if not (PROJECT_RAW / "train_transaction.csv").exists():
        pytest.skip("full dataset not present in data/raw/")
    return PROJECT_RAW
