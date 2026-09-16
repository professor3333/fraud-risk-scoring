from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud.data.load import load_identity, load_train, load_transactions
from fraud.data.split import load_split_config, split
from fraud.features.columns import load_feature_spec
from fraud.pipeline.build import build_pipeline
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.app import create_app
from fraud.serve.parity import choose_sample, freeze

FIXTURE_RAW = Path(__file__).resolve().parent / "fixtures" / "raw"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_RAW = PROJECT_ROOT / "data" / "raw"


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


# --- a calibrated model trained on the fixture, served from a temporary root ---------


@pytest.fixture(scope="session")
def served(fixture_raw_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A calibrated model trained on the fixture, saved under a temporary serving root."""
    root = tmp_path_factory.mktemp("serving")
    (root / "configs").mkdir()
    (root / "models").mkdir()
    df = load_train(fixture_raw_dir)
    parts = split(df, load_split_config(PROJECT_ROOT / "configs" / "split.yaml"))
    spec = load_feature_spec(PROJECT_ROOT / "configs" / "features" / "v2_freq.yaml")
    pipe = build_pipeline(spec, {"type": "xgboost", "params": {"n_estimators": 30}}, seed=0)
    pipe.fit(parts["train"], parts["train"][spec.target])
    held = parts["validation"]
    model = CalibratedModel(pipe, method="sigmoid").fit_calibrator(
        pipe.predict_proba(held)[:, 1], held[spec.target]
    )
    joblib.dump(model, root / "models" / "m.joblib")
    freeze(model, root / "models" / "m.joblib", choose_sample(parts["test"], n_per_group=5))
    cfg = root / "configs" / "serving.yaml"
    cfg.write_text(
        "model_path: models/m.joblib\n"
        "audit_db: models/audit.sqlite\n"
        "model_version: fixture-model\nbands: {review: 0.062, block: 0.42}\n"
        "model_info: {model: xgboost, experiment: fixture, feature_set: v2_freq, "
        "primary_metric: pr_auc, validation_pr_auc: 0.5, test_pr_auc: 0.4, calibration: sigmoid, "
        "training_window_days: [1, 122], validation_window_days: [123, 152]}\n"
    )
    return {
        "config": cfg,
        "model": model,
        "rows": parts["test"],
        "audit_db": root / "models" / "audit.sqlite",
    }


@pytest.fixture(scope="session")
def client(served: dict[str, Any]) -> Iterator[TestClient]:
    with TestClient(create_app(served["config"])) as c:
        yield c
