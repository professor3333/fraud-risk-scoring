"""ML-specific contract tests that are only implicit elsewhere, made explicit.

Each test names one property from the checklist in docs/testing.md. Fixture
only: no data, no network, seconds.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud.data import schema
from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.features.columns import load_feature_spec
from fraud.pipeline.build import build_pipeline
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.app import create_app

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
XGB = {"type": "xgboost", "params": {"n_estimators": 20, "max_depth": 3}}


@pytest.fixture(scope="module")
def fitted(fixture_raw_dir: Path) -> dict:
    parts = split(load_train(fixture_raw_dir), load_split_config(CONFIGS / "split.yaml"))
    spec = load_feature_spec(CONFIGS / "features" / "f5_interactions.yaml")
    pipe = build_pipeline(spec, XGB, seed=0)
    pipe.fit(parts["train"], parts["train"][spec.target])
    return {"parts": parts, "spec": spec, "pipe": pipe}


# --- split ordering, stated as two separate facts -----------------------------------


def test_train_is_strictly_before_validation(fitted: dict) -> None:
    p = fitted["parts"]
    assert p["train"][schema.TIME_COL].max() < p["validation"][schema.TIME_COL].min()


def test_validation_is_strictly_before_test(fitted: dict) -> None:
    p = fitted["parts"]
    assert p["validation"][schema.TIME_COL].max() < p["test"][schema.TIME_COL].min()


# --- no target (or time / id) in the fitted pipeline's inputs ---------------------------


def test_fitted_pipeline_never_reads_target_time_or_id(fitted: dict) -> None:
    ct = fitted["pipe"].named_steps["features"]
    consumed = {c for name, _, cols in ct.transformers_ if name != "remainder" for c in cols}
    dropped = {c for name, _, cols in ct.transformers_ if name == "remainder" for c in cols}
    assert schema.TARGET_COL in dropped and schema.TIME_COL in dropped
    assert schema.TARGET_COL not in consumed
    assert schema.ID_COL not in consumed
    assert schema.TIME_COL not in consumed
    # and the pipeline keeps working when those columns are absent from the input entirely
    rows = fitted["parts"]["validation"].drop(columns=[schema.TARGET_COL]).head(5)
    assert fitted["pipe"].predict_proba(rows).shape == (5, 2)


# --- feature count: fixed after fit, identical for every window ---------------------------


def test_feature_count_is_fixed_after_fit(fitted: dict) -> None:
    pipe, spec, parts = fitted["pipe"], fitted["spec"], fitted["parts"]
    names = pipe.named_steps["features"].get_feature_names_out()
    n_out = len(names)
    onehot = pipe.named_steps["features"].named_transformers_["cat"]["onehot"]
    n_onehot = sum(len(c) for c in onehot.categories_)
    assert n_out == len(spec.numeric) + n_onehot + len(spec.frequency)
    for window in ("train", "validation", "test"):
        assert pipe[:-1].transform(parts[window]).shape[1] == n_out
    assert pipe.named_steps["model"].n_features_in_ == n_out


# --- prediction shape and range ------------------------------------------------------------


def test_prediction_shape_and_range(fitted: dict) -> None:
    pipe, parts = fitted["pipe"], fitted["parts"]
    for window in ("validation", "test"):
        proba = pipe.predict_proba(parts[window])
        assert proba.shape == (len(parts[window]), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert proba.min() >= 0.0 and proba.max() <= 1.0
        assert not np.isnan(proba).any()
    calibrated = CalibratedModel(pipe, method="sigmoid").fit_calibrator(
        pipe.predict_proba(parts["validation"])[:, 1], parts["validation"][schema.TARGET_COL]
    )
    p = calibrated.predict_proba(parts["test"])[:, 1]
    assert p.shape == (len(parts["test"]),) and p.min() >= 0.0 and p.max() <= 1.0
    assert (
        calibrated.predict(parts["test"], threshold=0.5).tolist() == (p >= 0.5).astype(int).tolist()
    )


# --- serialization: bytes -> same object -> same numbers, twice -------------------------------


def test_model_serialization_roundtrip_is_exact(fitted: dict, tmp_path: Path) -> None:
    pipe, parts = fitted["pipe"], fitted["parts"]
    before = pipe.predict_proba(parts["test"])[:, 1]
    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path)
    once = joblib.load(path)
    joblib.dump(once, tmp_path / "again.joblib")
    twice = joblib.load(tmp_path / "again.joblib")
    np.testing.assert_array_equal(once.predict_proba(parts["test"])[:, 1], before)
    np.testing.assert_array_equal(twice.predict_proba(parts["test"])[:, 1], before)
    assert joblib.load(path).named_steps["features"].get_feature_names_out().tolist() == (
        pipe.named_steps["features"].get_feature_names_out().tolist()
    )


# --- API contract: routes, response fields, status codes -----------------------------------------


def test_api_contract_via_openapi(fixture_raw_dir: Path, tmp_path: Path) -> None:
    from fraud.serve.parity import choose_sample, freeze

    parts = split(load_train(fixture_raw_dir), load_split_config(CONFIGS / "split.yaml"))
    spec = load_feature_spec(CONFIGS / "features" / "v2_freq.yaml")
    pipe = build_pipeline(spec, XGB, seed=0).fit(parts["train"], parts["train"][spec.target])
    held = parts["validation"]
    model = CalibratedModel(pipe, method="sigmoid").fit_calibrator(
        pipe.predict_proba(held)[:, 1], held[spec.target]
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "models").mkdir()
    artifact = tmp_path / "models" / "m.joblib"
    joblib.dump(model, artifact)
    freeze(model, artifact, choose_sample(parts["test"], 3))
    (tmp_path / "configs" / "threshold.yaml").write_text(
        "costs:\n  false_negative: {fixed: 15.0, amount_coef: 1.0}\n"
        "  false_positive: {fixed: 2.0, amount_coef: 0.10}\n"
        "sweep: {start: 0.01, stop: 0.99, step: 0.01}\nthreshold: 0.08\n"
    )
    cfg = tmp_path / "configs" / "serving.yaml"
    cfg.write_text(
        "model_path: models/m.joblib\nthreshold_config: configs/threshold.yaml\n"
        "model_version: contract\nbands: {review: 0.062, block: 0.42}\n"
    )
    with TestClient(create_app(cfg)) as client:
        spec_doc = client.get("/openapi.json").json()
        paths = spec_doc["paths"]
        assert {"/health", "/model-info", "/predict", "/predict/batch", "/predict/csv"} <= set(
            paths
        )
        assert "get" in paths["/health"] and "post" in paths["/predict"]
        schemas = spec_doc["components"]["schemas"]
        pred = schemas["PredictionResponse"]
        assert set(pred["required"]) >= {
            "transaction_id", "fraud_probability", "decision", "risk_level", "action",
            "threshold", "model_version",
        }  # fmt: skip
        assert pred["properties"]["fraud_probability"]["minimum"] == 0.0
        assert pred["properties"]["fraud_probability"]["maximum"] == 1.0
        assert set(pred["properties"]["action"]["enum"]) == {"approve", "review", "block"}
        assert schemas["TransactionRequest"]["additionalProperties"] is False
        assert set(schemas["TransactionRequest"]["required"]) == {
            "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1",
        }  # fmt: skip
        # status codes: valid -> 200, invalid -> 422, unknown route -> 404
        row = parts["test"].iloc[0]
        payload = {
            "TransactionID": int(row[schema.ID_COL]), "TransactionDT": int(row[schema.TIME_COL]),
            "TransactionAmt": float(row["TransactionAmt"]), "ProductCD": str(row["ProductCD"]),
            "card1": float(row["card1"]),
        }  # fmt: skip
        assert client.post("/predict", json=payload).status_code == 200
        assert client.post("/predict", json={**payload, "TransactionAmt": 0}).status_code == 422
        assert client.get("/nope").status_code == 404
