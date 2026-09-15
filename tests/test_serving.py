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
    assert "threshold" not in body and "decision" not in body


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
        assert body["action"] == (
            "block" if expected >= 0.42 else "review" if expected >= 0.062 else "approve"
        )
        assert "decision" not in body and "threshold" not in body


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
    assert "Fraud review queue" in r.text and "/predict/csv" in r.text


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
    # single /predict uses the fixed bands; the batch matches it only under policy=threshold
    r = client.post("/predict/batch", json={"transactions": payloads, "policy": "threshold"})
    for x in r.json()["ranked"]:
        assert x["action"] == singles[x["transaction_id"]]["action"]
    assert r.json()["policy"]["policy"] == "threshold"


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
    assert "threshold" not in body
    assert body["bands"] == {"review": 0.062, "block": 0.42}
    assert body["n_inputs"] > 400 and body["parity_rows"] == 10


def test_csv_upload_scores_ranks_and_summarises(client: TestClient, served: dict[str, Any]) -> None:
    rows: pd.DataFrame = served["rows"].head(30)
    csv = rows.drop(columns=[schema.TARGET_COL, schema.HAS_IDENTITY_COL]).assign(extra_col=1)
    r = client.post("/predict/csv", files={"file": ("q.csv", csv.to_csv(index=False), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    s = body["summary"]
    assert s["analysed"] == 30 and s["ignored_columns"] == ["extra_col"]
    assert s["flagged"] == s["review"] + s["high_risk"]
    assert s["approve"] + s["flagged"] == 30
    probs = [x["fraud_probability"] for x in body["rows"]]
    assert probs == sorted(probs, reverse=True)
    assert [x["rank"] for x in body["rows"]] == list(range(1, 31))
    # CSV path == request path, row by row (the two frame builders agree)
    single = {
        p["TransactionID"]: client.post("/predict", json=p).json()
        for p in (_payload(row) for _, row in rows.iterrows())
    }
    for x in body["rows"]:
        assert x["fraud_probability"] == pytest.approx(
            single[x["transaction_id"]]["fraud_probability"], abs=1e-9
        )
        assert "TransactionAmt" in x["details"] and "isFraud" not in x["details"]


def test_csv_upload_rejects_bad_files(client: TestClient, served: dict[str, Any]) -> None:
    rows: pd.DataFrame = served["rows"].head(3).drop(columns=[schema.TARGET_COL])
    r = client.post("/predict/csv", files={"file": ("q.csv", "", "text/csv")})
    assert r.status_code == 422
    no_amount = rows.drop(columns=["TransactionAmt"]).to_csv(index=False)
    r = client.post("/predict/csv", files={"file": ("q.csv", no_amount, "text/csv")})
    assert r.status_code == 422 and "TransactionAmt" in r.text
    bad_number = rows.astype("str")
    bad_number.loc[bad_number.index[0], "card2"] = "abc"
    text = bad_number.to_csv(index=False)
    r = client.post("/predict/csv", files={"file": ("q.csv", text, "text/csv")})
    assert r.status_code == 422 and "card2" in r.text


def test_dashboard_and_sample_are_served(client: TestClient) -> None:
    assert "Fraud review queue" in client.get("/").text
    assert "/predict" in client.get("/single").text
    r = client.get("/sample.csv")
    assert r.status_code == 200 and r.text.startswith("TransactionID,")
    scored = client.post("/predict/csv", files={"file": ("s.csv", r.content, "text/csv")})
    assert scored.status_code == 200 and scored.json()["summary"]["analysed"] == 200


def test_batch_rank_policy_reviews_exactly_the_budget(
    client: TestClient, served: dict[str, Any]
) -> None:
    """Block by threshold, review the top-N remaining, approve the rest (docs/review_policy.md)."""
    rows: pd.DataFrame = served["rows"].head(40)
    payloads = [_payload(row) for _, row in rows.iterrows()]
    for budget in (0, 5, 1000):
        body = client.post(
            "/predict/batch", json={"transactions": payloads, "review_budget": budget}
        ).json()
        ranked = body["ranked"]
        n_block = sum(x["action"] == "block" for x in ranked)
        n_review = sum(x["action"] == "review" for x in ranked)
        assert n_block == sum(x["fraud_probability"] >= 0.42 for x in ranked)
        assert n_review == min(budget, len(ranked) - n_block)
        # reviewed rows are exactly the highest-probability non-blocked rows
        non_blocked = [x for x in ranked if x["action"] != "block"]  # already sorted desc
        assert [x["action"] for x in non_blocked] == ["review"] * n_review + ["approve"] * (
            len(non_blocked) - n_review
        )
        pol = body["policy"]
        assert pol["policy"] == "rank" and pol["review_budget"] == budget
        if n_review:
            assert pol["review_cutoff"] == pytest.approx(
                min(x["fraud_probability"] for x in non_blocked[:n_review])
            )
        else:
            assert pol["review_cutoff"] is None
        assert body["counts"] == {"block": n_block, "review": n_review,
                                  "approve": len(ranked) - n_block - n_review}  # fmt: skip
    # default budget comes from the server config when the request omits it
    body = client.post("/predict/batch", json={"transactions": payloads}).json()
    assert body["policy"]["review_budget"] == 200
    assert client.get("/model-info").json()["default_review_budget"] == 200
    assert client.get("/model-info").json()["default_policy"] == "rank"


def test_csv_upload_honours_review_budget(client: TestClient, served: dict[str, Any]) -> None:
    rows: pd.DataFrame = served["rows"].head(30)
    csv = rows.drop(columns=[schema.TARGET_COL, schema.HAS_IDENTITY_COL]).to_csv(index=False)
    files = {"file": ("q.csv", csv, "text/csv")}
    body = client.post("/predict/csv?review_budget=7", files=files).json()
    s = body["summary"]
    assert s["policy"]["policy"] == "rank" and s["policy"]["review_budget"] == 7
    assert s["review"] == min(7, 30 - s["high_risk"])
    assert s["flagged"] == s["review"] + s["high_risk"]
    body_t = client.post("/predict/csv?policy=threshold", files=files).json()
    assert body_t["summary"]["policy"]["policy"] == "threshold"
    assert body_t["summary"]["policy"]["review_threshold"] == 0.062


# --- prediction audit trail --------------------------------------------------------------


def test_predictions_are_audited(client: TestClient, served: dict[str, Any]) -> None:
    import sqlite3

    before = client.get("/health").json()["audit_events"]
    assert before is not None
    payload = _payload(served["rows"].iloc[0])
    r = client.post("/predict", json=payload, headers={"X-Request-ID": "req-abc"})
    assert r.headers["X-Request-ID"] == "req-abc"
    single = r.json()
    rows = [_payload(row) for _, row in served["rows"].head(6).iterrows()]
    rb = client.post("/predict/batch", json={"transactions": rows, "review_budget": 2})
    batch_rid = rb.headers["X-Request-ID"]
    assert len(batch_rid) == 32  # generated when the caller sends none
    assert client.get("/health").json()["audit_events"] == before + 1 + 6

    recent = client.get("/audit/recent?limit=10").json()
    assert len(recent) >= 7
    batch_events = [e for e in recent if e["request_id"] == batch_rid]
    assert len(batch_events) == 6 and {e["batch_size"] for e in batch_events} == {6}
    assert {e["policy"] for e in batch_events} == {"rank"}
    assert {e["review_budget"] for e in batch_events} == {2}
    assert sum(e["action"] == "review" for e in batch_events) == min(
        2, 6 - sum(e["action"] == "block" for e in batch_events)
    )
    mine = [e for e in recent if e["request_id"] == "req-abc"]
    assert len(mine) == 1
    e = mine[0]
    assert e["endpoint"] == "/predict" and e["policy"] == "threshold"
    assert e["transaction_id"] == single["transaction_id"]
    assert e["fraud_probability"] == pytest.approx(single["fraud_probability"])
    assert e["action"] == single["action"] and e["model_version"] == single["model_version"]
    assert e["block_threshold"] == 0.42 and e["review_threshold"] == 0.062
    assert e["latency_ms"] > 0 and e["scored_at"].endswith("+00:00")
    # per-transaction lookup: what did we ever say about this transaction?
    by_txn = client.get(f"/audit/recent?transaction_id={single['transaction_id']}").json()
    assert all(x["transaction_id"] == single["transaction_id"] for x in by_txn) and by_txn
    # the file is a plain SQLite database an investigator can open directly
    db = sqlite3.connect(served["audit_db"])
    n = db.execute("SELECT COUNT(*) FROM prediction_events").fetchone()[0]
    assert n == before + 7


def test_csv_upload_is_audited_with_one_request_id(
    client: TestClient, served: dict[str, Any]
) -> None:
    rows: pd.DataFrame = served["rows"].head(15)
    csv = rows.drop(columns=[schema.TARGET_COL, schema.HAS_IDENTITY_COL]).to_csv(index=False)
    r = client.post("/predict/csv?review_budget=3", files={"file": ("q.csv", csv, "text/csv")})
    rid = r.headers["X-Request-ID"]
    events = [e for e in client.get("/audit/recent?limit=50").json() if e["request_id"] == rid]
    assert len(events) == 15 and {e["endpoint"] for e in events} == {"/predict/csv"}
    assert {e["review_budget"] for e in events} == {3}


def test_audit_can_be_disabled(
    fixture_raw_dir: Path, served: dict[str, Any], tmp_path: Path
) -> None:
    cfg = tmp_path / "serving.yaml"
    text = (
        Path(served["config"])
        .read_text()
        .replace(
            "model_path: models/m.joblib",
            f"model_path: {Path(served['config']).parents[1] / 'models' / 'm.joblib'}",
        )
    )
    cfg.write_text(text + "audit_db: null\n")
    with TestClient(create_app(cfg)) as c:
        assert c.get("/health").json()["audit_events"] is None
        assert c.get("/audit/recent").status_code == 404
        assert c.post("/predict", json=_payload(served["rows"].iloc[0])).status_code == 200
