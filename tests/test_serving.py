"""Serving tests: bad input rejected, /predict matches the offline pipeline, /health cheap."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud.data import schema
from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.features.columns import load_feature_spec
from fraud.pipeline.build import build_pipeline
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.app import create_app, request_to_frame
from fraud.serve.parity import choose_sample, freeze
from fraud.serve.schemas import IDENTITY_FIELDS

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.08


@pytest.fixture(scope="module")
def served(fixture_raw_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A calibrated model trained on the fixture, saved under a temporary serving root."""
    root = tmp_path_factory.mktemp("serving")
    (root / "configs").mkdir()
    (root / "models").mkdir()
    df = load_train(fixture_raw_dir)
    parts = split(df, load_split_config(ROOT / "configs" / "split.yaml"))
    spec = load_feature_spec(ROOT / "configs" / "features" / "v2_freq.yaml")
    pipe = build_pipeline(spec, {"type": "xgboost", "params": {"n_estimators": 30}}, seed=0)
    pipe.fit(parts["train"], parts["train"][spec.target])
    held = parts["validation"]
    model = CalibratedModel(pipe, method="sigmoid").fit_calibrator(
        pipe.predict_proba(held)[:, 1], held[spec.target]
    )
    joblib.dump(model, root / "models" / "m.joblib")
    freeze(model, root / "models" / "m.joblib", choose_sample(parts["test"], n_per_group=5))
    (root / "configs" / "threshold.yaml").write_text(
        "costs:\n  false_negative: {fixed: 15.0, amount_coef: 1.0}\n"
        "  false_positive: {fixed: 2.0, amount_coef: 0.10}\n"
        "sweep: {start: 0.01, stop: 0.99, step: 0.01}\n"
        f"threshold: {THRESHOLD}\n"
    )
    cfg = root / "configs" / "serving.yaml"
    cfg.write_text(
        "model_path: models/m.joblib\nthreshold_config: configs/threshold.yaml\n"
        "model_version: fixture-model\nbands: {review: 0.062, block: 0.42}\n"
        "model_info: {model: xgboost, experiment: fixture, feature_set: v2_freq, "
        "primary_metric: pr_auc, validation_pr_auc: 0.5, test_pr_auc: 0.4, calibration: sigmoid, "
        "training_window_days: [1, 122], validation_window_days: [123, 152]}\n"
    )
    return {"config": cfg, "model": model, "rows": parts["test"]}


@pytest.fixture(scope="module")
def client(served: dict[str, Any]) -> Iterator[TestClient]:
    with TestClient(create_app(served["config"])) as c:
        yield c


def _payload(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col, val in row.items():
        if col in (schema.TARGET_COL, schema.HAS_IDENTITY_COL):
            continue
        if pd.isna(val):
            continue
        if col in (schema.ID_COL, schema.TIME_COL):
            out[col] = int(val)
        elif col in schema.TRANSACTION_STR_COLS | schema.IDENTITY_STR_COLS:
            out[col] = str(val)
        else:
            out[col] = float(val)
    return out


def test_health_is_cheap_and_reports_version(client: TestClient) -> None:
    t0 = time.perf_counter()
    r = client.get("/health")
    assert time.perf_counter() - t0 < 0.5
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("fixture-model@")
    assert body["threshold"] == THRESHOLD


def test_predict_matches_offline_pipeline(client: TestClient, served: dict[str, Any]) -> None:
    """G8: the API returns the same probability as the offline object on the same raw rows."""
    rows: pd.DataFrame = served["rows"]
    offline = served["model"].predict_proba(rows)[:, 1]
    with_id = rows[rows[schema.HAS_IDENTITY_COL]].head(5)
    without_id = rows[~rows[schema.HAS_IDENTITY_COL]].head(5)
    sample = pd.concat([with_id, without_id])
    assert len(sample) == 10
    expected_scores = offline[sample.index.map(rows.index.get_loc)]
    for (_, row), expected in zip(sample.iterrows(), expected_scores, strict=True):
        r = client.post("/predict", json=_payload(row))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["transaction_id"] == int(row[schema.ID_COL])
        assert body["fraud_probability"] == pytest.approx(expected, abs=1e-9)
        assert body["decision"] == ("decline" if expected >= THRESHOLD else "approve")
        assert body["threshold"] == THRESHOLD


def test_request_to_frame_reconstructs_has_identity(served: dict[str, Any]) -> None:
    rows: pd.DataFrame = served["rows"]
    with_id = rows[rows[schema.HAS_IDENTITY_COL]].iloc[0]
    without_id = rows[~rows[schema.HAS_IDENTITY_COL]].iloc[0]
    assert request_to_frame(_payload(with_id))[schema.HAS_IDENTITY_COL].item() is True
    assert request_to_frame(_payload(without_id))[schema.HAS_IDENTITY_COL].item() is False
    frame = request_to_frame(_payload(without_id))
    assert frame[list(IDENTITY_FIELDS)].isna().all(axis=None)


@pytest.mark.parametrize(
    ("mutate", "detail_contains"),
    [
        (lambda p: p.pop("TransactionAmt"), "TransactionAmt"),
        (lambda p: p.update(TransactionAmt=-5.0), "greater than 0"),
        (lambda p: p.update(ProductCD=12), "string"),
        (lambda p: p.update(not_a_column=1), "extra"),
        (lambda p: p.pop("card1"), "card1"),
    ],
)
def test_bad_input_is_rejected_with_a_useful_body(
    client: TestClient, served: dict[str, Any], mutate: Any, detail_contains: str
) -> None:
    payload = _payload(served["rows"].iloc[0])
    mutate(payload)
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert detail_contains.lower() in r.text.lower()


def test_index_page_is_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Fraud Risk Scoring" in r.text and "/predict" in r.text


@pytest.mark.slow
def test_real_model_parity_on_real_rows(full_raw_dir: Path) -> None:
    """The deployed artifact scores real validation rows identically via the API and offline."""
    cfg = ROOT / "configs" / "serving.yaml"
    model_path = ROOT / "models" / "xgb_f5_capacity_calibrated.joblib"
    if not model_path.exists():
        pytest.skip("served artifact not built")
    df = load_train(full_raw_dir, cache_dir=full_raw_dir.parent / "processed")
    val = split(df, load_split_config(ROOT / "configs" / "split.yaml"))["validation"]
    sample = pd.concat(
        [val[val[schema.HAS_IDENTITY_COL]].head(10), val[~val[schema.HAS_IDENTITY_COL]].head(10)]
    )
    offline = joblib.load(model_path).predict_proba(sample)[:, 1]
    with TestClient(create_app(cfg)) as c:
        for (_, row), expected in zip(sample.iterrows(), offline, strict=True):
            body = c.post("/predict", json=_payload(row)).json()
            assert body["fraud_probability"] == pytest.approx(expected, abs=1e-9)


def test_predict_returns_risk_level_and_action(client: TestClient, served: dict[str, Any]) -> None:
    r = client.post("/predict", json=_payload(served["rows"].iloc[0])).json()
    assert r["risk_level"] in ("low", "medium", "high")
    assert r["action"] in ("approve", "review", "block")
    p = r["fraud_probability"]
    expected = "block" if p >= 0.42 else "review" if p >= 0.062 else "approve"
    assert r["action"] == expected


def test_batch_is_ranked_and_matches_single_predictions(
    client: TestClient, served: dict[str, Any]
) -> None:
    rows: pd.DataFrame = served["rows"].head(12)
    payloads = [_payload(row) for _, row in rows.iterrows()]
    r = client.post("/predict/batch", json={"transactions": payloads})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 12 and [x["rank"] for x in body["ranked"]] == list(range(1, 13))
    probs = [x["fraud_probability"] for x in body["ranked"]]
    assert probs == sorted(probs, reverse=True)
    assert sum(body["counts"].values()) == 12
    singles = {p["TransactionID"]: client.post("/predict", json=p).json() for p in payloads}
    for x in body["ranked"]:
        s = singles[x["transaction_id"]]
        assert x["fraud_probability"] == pytest.approx(s["fraud_probability"], abs=1e-9)
        assert x["action"] == s["action"]


def test_batch_rejects_empty_and_bad_rows(client: TestClient, served: dict[str, Any]) -> None:
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
    bad = _payload(served["rows"].iloc[0])
    bad.pop("TransactionAmt")
    r = client.post("/predict/batch", json={"transactions": [bad]})
    assert r.status_code == 422 and "TransactionAmt" in r.text


def test_model_info(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "xgboost" and body["primary_metric"] == "pr_auc"
    assert body["version"].startswith("fixture-model@")
    assert body["threshold"] == THRESHOLD
    assert body["bands"] == {"review": 0.062, "block": 0.42}
    assert body["n_inputs"] > 400 and body["parity_rows"] == 10
