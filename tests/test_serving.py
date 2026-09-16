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


def test_payloads_to_frame_equals_row_by_row_construction(served: dict[str, Any]) -> None:
    """The vectorised batch frame is the single-request frame, row for row, dtype for dtype
    (the golden and the batch == single test both rest on this)."""
    from fraud.serve.frames import payloads_to_frame, request_to_frame

    rows: pd.DataFrame = served["rows"].head(40)
    payloads = [_payload(row) for _, row in rows.iterrows()]
    payloads[3] = {k: v for k, v in payloads[3].items() if not k.startswith("id_")}  # no identity
    vectorised = payloads_to_frame(payloads)
    row_by_row = pd.concat([request_to_frame(p) for p in payloads], ignore_index=True)
    pd.testing.assert_frame_equal(vectorised, row_by_row, check_dtype=True, check_exact=True)
    assert not vectorised.loc[3, schema.HAS_IDENTITY_COL]


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


def test_csv_upload_limits_are_enforced_before_the_body_is_held(
    served: dict[str, Any], tmp_path: Path
) -> None:
    """A body over the byte cap is refused (declared or not) and a CSV over the row limit
    is refused without parsing past the limit."""
    from fraud.serve import app as serving_app

    cfg = tmp_path / "serving.yaml"
    text = (
        Path(served["config"])
        .read_text()
        .replace(
            "model_path: models/m.joblib",
            f"model_path: {Path(served['config']).parents[1] / 'models' / 'm.joblib'}",
        )
    )
    cfg.write_text(text + "audit_db: null\nmax_upload_bytes: 20000\n")
    rows: pd.DataFrame = served["rows"].head(3).drop(columns=[schema.TARGET_COL])
    small = rows.to_csv(index=False)
    with TestClient(create_app(cfg)) as c:
        ok = c.post("/predict/csv", files={"file": ("q.csv", small, "text/csv")})
        assert ok.status_code == 200
        big = small + ("x," * 200 + "\n") * 200  # > 20 kB of junk after valid rows
        r = c.post("/predict/csv", files={"file": ("q.csv", big, "text/csv")})
        assert r.status_code == 413 and "20000" in r.text
        # a client lying about (or omitting) the length is caught by the streamed read
        r = c.post(
            "/predict/csv", files={"file": ("q.csv", big, "text/csv")},
            headers={"content-length": "100"},
        )  # fmt: skip
        assert r.status_code == 413
    # the row limit: MAX_CSV_ROWS + 1 rows are read, no more, and the upload is refused
    limit = serving_app.MAX_CSV_ROWS
    header = ",".join(rows.columns) + "\n"
    one = rows.iloc[[0]].to_csv(index=False, header=False).strip()
    body = header + "\n".join(one for _ in range(limit + 5)) + "\n"
    calls: list[int | None] = []
    original = pd.read_csv

    def spy(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs.get("nrows"))
        return original(*args, **kwargs)

    serving_app.pd.read_csv = spy  # type: ignore[assignment]
    try:
        with TestClient(create_app(Path(served["config"]))) as c:
            r = c.post("/predict/csv", files={"file": ("q.csv", body, "text/csv")})
    finally:
        serving_app.pd.read_csv = original  # type: ignore[assignment]
    assert r.status_code == 422 and f"at most {limit} rows" in r.text
    assert calls == [limit + 1]


def test_dashboard_and_sample_are_served(client: TestClient) -> None:
    assert "Fraud review queue" in client.get("/").text
    assert "/predict" in client.get("/single").text
    r = client.get("/sample.csv")
    assert r.status_code == 200 and r.text.startswith("TransactionID,")
    scored = client.post("/predict/csv", files={"file": ("s.csv", r.content, "text/csv")})
    assert scored.status_code == 200 and scored.json()["summary"]["analysed"] == 200


def _fresh_client(served: dict[str, Any], tmp_path: Path, audit: bool = True) -> TestClient:
    """A service on its own audit database, so budget accounting starts from zero."""
    cfg = tmp_path / "serving.yaml"
    text = (
        Path(served["config"])
        .read_text()
        .replace(
            "model_path: models/m.joblib",
            f"model_path: {Path(served['config']).parents[1] / 'models' / 'm.joblib'}",
        )
    )
    db = f"audit_db: {tmp_path / 'audit.sqlite'}" if audit else "audit_db: null"
    cfg.write_text(text + db + "\n")
    return TestClient(create_app(cfg))


def _day(payload: dict[str, Any]) -> int:
    return int(payload[schema.TIME_COL]) // 86_400


def _released(payloads: list[dict[str, Any]], budget: int) -> dict[int, int]:
    """Reviews the day has released by its latest transaction in the request."""
    import math

    latest: dict[int, int] = {}
    for p in payloads:
        latest[_day(p)] = max(latest.get(_day(p), 0), int(p[schema.TIME_COL]))
    return {d: math.ceil(budget * ((t - d * 86_400 + 1) / 86_400)) for d, t in latest.items()}


def _expected_reviews(
    ranked: list[dict[str, Any]], payloads: list[dict[str, Any]], budget: int
) -> int:
    """The rank policy per transaction day: min(released, non-blocked rows) summed."""
    day_of = {p["TransactionID"]: _day(p) for p in payloads}
    released = _released(payloads, budget)
    by_day: dict[int, int] = {}
    for x in ranked:
        if x["action"] != "block":
            by_day[day_of[x["transaction_id"]]] = by_day.get(day_of[x["transaction_id"]], 0) + 1
    return sum(min(released[d], n) for d, n in by_day.items())


def test_batch_rank_policy_reviews_the_budget_per_transaction_day(
    served: dict[str, Any], tmp_path: Path
) -> None:
    """Block by threshold, review the top-N remaining *per day*, approve the rest."""
    rows: pd.DataFrame = served["rows"].head(40)
    payloads = [_payload(row) for _, row in rows.iterrows()]
    day_of = {p["TransactionID"]: _day(p) for p in payloads}
    n_days = len(set(day_of.values()))
    assert n_days > 1  # the fixture rows span days; the policy must not pool them
    with _fresh_client(served, tmp_path) as client:
        for budget in (0, 1, 1000):
            body = client.post(
                "/predict/batch", json={"transactions": payloads, "review_budget": budget}
            ).json()
            ranked = body["ranked"]
            n_block = sum(x["action"] == "block" for x in ranked)
            n_review = sum(x["action"] == "review" for x in ranked)
            assert n_block == sum(x["fraud_probability"] >= 0.42 for x in ranked)
            assert n_review == _expected_reviews(ranked, payloads, budget)
            # within each day the reviewed rows are the highest-probability non-blocked ones
            for day in set(day_of.values()):
                todays = [
                    x
                    for x in ranked
                    if day_of[x["transaction_id"]] == day and x["action"] != "block"
                ]  # already sorted desc
                k = sum(x["action"] == "review" for x in todays)
                assert [x["action"] for x in todays] == ["review"] * k + ["approve"] * (
                    len(todays) - k
                )
            pol = body["policy"]
            assert pol["policy"] == "rank" and pol["review_budget"] == budget
            assert pol["budget_accounting"] == "audit_trail"
            # re-scoring the same rows: everything the days have released by these rows
            assert pol["review_capacity"] == sum(_released(payloads, budget).values())
            assert pol["review_capacity"] <= budget * n_days
            if n_review:
                reviewed = [x["fraud_probability"] for x in ranked if x["action"] == "review"]
                assert pol["review_cutoff"] == pytest.approx(min(reviewed))
            else:
                assert pol["review_cutoff"] is None
            assert body["counts"] == {"block": n_block, "review": n_review,
                                      "approve": len(ranked) - n_block - n_review}  # fmt: skip
        # default budget comes from the server config when the request omits it
        body = client.post("/predict/batch", json={"transactions": payloads}).json()
        assert body["policy"]["review_budget"] == 200
        assert client.get("/model-info").json()["default_review_budget"] == 200
        assert client.get("/model-info").json()["default_policy"] == "rank"


def test_review_budget_is_shared_across_requests_of_the_same_day(
    served: dict[str, Any], tmp_path: Path
) -> None:
    """Ten small batches of one day may not review ten budgets; re-sending rows is a
    re-decision, not a second review; other days have their own budget; no audit, no memory."""
    rows: pd.DataFrame = served["rows"]
    payloads = [_payload(row) for _, row in rows.iterrows()]
    by_day: dict[int, list[dict[str, Any]]] = {}
    for p in payloads:
        by_day.setdefault(_day(p), []).append(p)
    day, todays = max(by_day.items(), key=lambda kv: len(kv[1]))
    assert len(todays) >= 4
    first, second = todays[: len(todays) // 2], todays[len(todays) // 2 :]
    other_day = next(d for d in by_day if d != day)

    def reviews(client: TestClient, batch: list[dict[str, Any]], budget: int) -> tuple[int, int]:
        body = client.post(
            "/predict/batch", json={"transactions": batch, "review_budget": budget}
        ).json()
        c = body["counts"]
        assert c["review"] == min(body["policy"]["review_capacity"], c["review"] + c["approve"])
        return c["review"], body["policy"]["review_capacity"]

    with _fresh_client(served, tmp_path) as client:
        # a budget of 1: the first batch spends it, the second gets nothing
        n1, cap1 = reviews(client, first, 1)
        assert cap1 == 1
        n2, cap2 = reviews(client, second, 1)
        assert cap2 == 1 - n1 and (n2 == 0 or not n1)
        # a different day is untouched
        _, cap_other = reviews(client, by_day[other_day], 1)
        assert cap_other == 1
        # re-sending the first batch: its own earlier reviews do not count against it
        n1_again, cap_again = reviews(client, first, 1)
        assert cap_again == 1 and n1_again == n1
        # raising the budget later: what the day has released by these rows, minus what stands
        released = _released(second, 3)[day]
        _, cap3 = reviews(client, second, 3)
        assert cap3 == max(released - n1, 0)
        # the single-prediction endpoint (threshold policy) re-decides its row and the
        # day's standing reviews — latest decision per transaction — are what is charged
        client.post("/predict", json=todays[0])
        standing = 0
        for p in first:
            latest = client.get(f"/audit/recent?transaction_id={p['TransactionID']}").json()[0]
            standing += latest["action"] == "review"
        _, cap4 = reviews(client, second, 3)
        assert cap4 == max(released - standing, 0)
    (tmp_path / "noaudit").mkdir(exist_ok=True)
    with _fresh_client(served, tmp_path / "noaudit", audit=False) as client:
        body = client.post(
            "/predict/batch", json={"transactions": first, "review_budget": 1}
        ).json()
        assert body["policy"]["budget_accounting"] == "per_request"
        body = client.post(
            "/predict/batch", json={"transactions": second, "review_budget": 1}
        ).json()
        assert body["policy"]["review_capacity"] == 1  # no memory of the first batch


def test_csv_upload_honours_review_budget(served: dict[str, Any], tmp_path: Path) -> None:
    rows: pd.DataFrame = served["rows"].head(30)
    payloads = [_payload(row) for _, row in rows.iterrows()]
    csv = rows.drop(columns=[schema.TARGET_COL, schema.HAS_IDENTITY_COL]).to_csv(index=False)
    files = {"file": ("q.csv", csv, "text/csv")}
    with _fresh_client(served, tmp_path) as client:
        body = client.post("/predict/csv?review_budget=7", files=files).json()
        s = body["summary"]
        assert s["policy"]["policy"] == "rank" and s["policy"]["review_budget"] == 7
        ranked = [
            {"transaction_id": r["transaction_id"], "action": r["action"]} for r in body["rows"]
        ]
        assert s["review"] == _expected_reviews(ranked, payloads, 7)
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
    assert sum(e["action"] == "review" for e in batch_events) <= 2 * len(
        {e["transaction_dt"] // 86_400 for e in batch_events}
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
    # every event carries the transaction's own clock, so a delayed label can be aged later
    by_id = {e["transaction_id"]: e["transaction_dt"] for e in events}
    assert by_id == dict(zip(rows[schema.ID_COL], rows[schema.TIME_COL], strict=True))


def test_outcomes_attach_to_scored_transactions(client: TestClient, served: dict[str, Any]) -> None:
    import sqlite3

    rows: pd.DataFrame = served["rows"].head(3)
    for _, row in rows.iterrows():
        assert client.post("/predict", json=_payload(row)).status_code == 200
    outcomes = [
        {
            "transaction_id": int(r[schema.ID_COL]),
            "is_fraud": bool(r[schema.TARGET_COL]),
            "event_dt": int(r[schema.TIME_COL]),
            "observed_dt": int(r[schema.TIME_COL]) + 30 * 86_400,
        }
        for _, r in rows.iterrows()
    ]
    r = client.post("/outcomes", json={"outcomes": outcomes, "source": "chargebacks"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["received"] == 3 and body["recorded"] == 3 and body["total"] >= 3
    # a second delivery of the same transactions changes nothing: first label wins
    again = client.post("/outcomes", json={"outcomes": outcomes}).json()
    assert again["recorded"] == 0 and again["total"] == body["total"]
    db = sqlite3.connect(served["audit_db"])
    stored = dict(db.execute("SELECT transaction_id, is_fraud FROM outcomes").fetchall())
    for o in outcomes:
        assert stored[o["transaction_id"]] == int(o["is_fraud"])
    assert db.execute("SELECT MAX(as_of_dt) FROM label_feed").fetchone()[0] == max(
        o["observed_dt"] for o in outcomes
    )
    # validation: an unknown field, an empty list, a negative clock
    bad = client.post("/outcomes", json={"outcomes": [{**outcomes[0], "note": "x"}]})
    assert bad.status_code == 422
    assert client.post("/outcomes", json={"outcomes": []}).status_code == 422
    assert (
        client.post(
            "/outcomes", json={"outcomes": [{**outcomes[0], "observed_dt": -1}]}
        ).status_code
        == 422
    )


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


def test_requests_and_inputs_are_recorded_for_monitoring(
    client: TestClient, served: dict[str, Any]
) -> None:
    import sqlite3

    db = sqlite3.connect(served["audit_db"])
    before_req = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    before_in = db.execute("SELECT COUNT(*) FROM input_features").fetchone()[0]
    rows = [_payload(row) for _, row in served["rows"].head(4).iterrows()]
    r = client.post("/predict/batch", json={"transactions": rows})
    rid = r.headers["X-Request-ID"]
    bad = client.post("/predict", json={"TransactionID": 1})  # 422, still a request row
    assert bad.status_code == 422
    req = db.execute(
        "SELECT endpoint, status_code, n_rows FROM requests ORDER BY id DESC LIMIT 2"
    ).fetchall()
    assert req[0] == ("/predict", 422, 0) and req[1] == ("/predict/batch", 200, 4)
    assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == before_req + 2
    assert db.execute("SELECT COUNT(*) FROM input_features").fetchone()[0] == before_in + 4
    snap = db.execute(
        "SELECT product, has_identity, hour, n_missing_transaction FROM input_features "
        "WHERE request_id = ?", (rid,)
    ).fetchall()  # fmt: skip
    assert len(snap) == 4 and all(0 <= h <= 23 for _, _, h, _ in snap)
