"""Serving tests: bad input rejected, /predict matches the offline pipeline, /health cheap."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from fraud.data import schema
from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.serve.app import create_app, request_to_frame
from fraud.serve.schemas import IDENTITY_FIELDS

from .conftest import ADMIN_KEY

ROOT = Path(__file__).resolve().parents[1]


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
        assert body["risk_level"] == (
            "high" if expected >= 0.42 else "medium" if expected >= 0.062 else "low"
        )
        # scoring-only: choosing review vs approve needs the day's other scores
        assert "action" not in body
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
    model_path = ROOT / yaml.safe_load(cfg.read_text())["model_path"]
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


def test_predict_scores_but_does_not_choose_an_action(
    client: TestClient, served: dict[str, Any]
) -> None:
    """The production policy blocks by threshold then reviews the day's highest
    remaining scores. The second half cannot be decided for one transaction, so
    /predict reports the band and leaves selection to the review queue."""
    r = client.post("/predict", json=_payload(served["rows"].iloc[0])).json()
    assert r["risk_level"] in ("low", "medium", "high")
    assert "action" not in r, "a single score cannot know the day's ranking"
    p = r["fraud_probability"]
    assert r["risk_level"] == ("high" if p >= 0.42 else "medium" if p >= 0.062 else "low")
    assert set(r) == {"transaction_id", "fraud_probability", "risk_level", "model_version"}


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
    # /predict returns no action, but the band it reports must agree with the batch's
    r = client.post("/predict/batch", json={"transactions": payloads, "policy": "threshold"})
    for x in r.json()["ranked"]:
        assert x["risk_level"] == singles[x["transaction_id"]]["risk_level"]
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


@contextmanager
def _fresh_client(
    served: dict[str, Any], tmp_path: Path, audit: bool = True, admin_key: str | None = ADMIN_KEY
) -> Iterator[TestClient]:
    """A service on its own audit database, so budget accounting starts from zero.

    With an admin key the client sends it by default; with None the admin endpoints are
    closed, as on the anonymous public demo.
    """
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
    with pytest.MonkeyPatch.context() as mp:  # the key is read at startup, inside the client
        if admin_key is None:
            mp.delenv("FRAUD_ADMIN_API_KEY", raising=False)
        else:
            mp.setenv("FRAUD_ADMIN_API_KEY", admin_key)
        headers = {"X-API-Key": admin_key} if admin_key else None
        with TestClient(create_app(cfg), headers=headers) as c:
            yield c


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
    assert e["model_version"] == single["model_version"]
    # the audit records the payment-path outcome, which for a lone transaction is block
    # or let-it-proceed; it never claims a review, so it never charges the day's budget
    assert e["action"] == ("block" if single["fraud_probability"] >= 0.42 else "approve")
    assert e["action"] != "review"
    assert e["block_threshold"] == 0.42 and e["review_threshold"] is None
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
    fixture_raw_dir: Path, served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setenv("FRAUD_ADMIN_API_KEY", ADMIN_KEY)
    with TestClient(create_app(cfg), headers={"X-API-Key": ADMIN_KEY}) as c:
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


# --- public-API hardening ----------------------------------------------------------------


def test_api_key_guards_scoring_and_labels_when_configured(
    served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(served["rows"].iloc[0])
    # open by default: /health says so and nothing needs a key
    with _fresh_client(served, tmp_path) as c:
        assert c.get("/health").json()["auth"] == "open"
        assert c.post("/predict", json=payload).status_code == 200
    monkeypatch.setenv("FRAUD_API_KEY", "s3cret")
    (tmp_path / "keyed").mkdir()
    with _fresh_client(served, tmp_path / "keyed") as c:
        assert c.get("/health").json()["auth"] == "api_key"
        assert c.get("/model-info").status_code == 200  # facts stay open
        assert c.get("/").status_code == 200
        for path, kwargs in (
            ("/predict", {"json": payload}),
            ("/predict/batch", {"json": {"transactions": [payload]}}),
            ("/explain", {"json": payload}),
        ):
            r = c.post(path, **kwargs)
            assert r.status_code == 401 and "X-API-Key" in r.json()["detail"], path
            assert c.post(path, headers={"X-API-Key": "wrong"}, **kwargs).status_code == 401
            assert c.post(path, headers={"X-API-Key": "s3cret"}, **kwargs).status_code == 200
        # the scoring key never unlocks the admin routes, nor the admin key the scoring ones
        assert c.get("/audit/recent", headers={"X-API-Key": "s3cret"}).status_code == 401
        assert c.get("/audit/recent", headers={"X-API-Key": ADMIN_KEY}).status_code == 200
        assert c.post("/predict", json=payload, headers={"X-API-Key": ADMIN_KEY}).status_code == 401
        assert c.post("/predict", json=payload, headers={"X-API-Key": "s3cret"}).status_code == 200
        # rejected calls are still recorded and carry a request id
        r = c.post("/predict", json=payload, headers={"X-Request-ID": "denied-1"})
        assert r.status_code == 401 and r.headers["X-Request-ID"] == "denied-1"
        import sqlite3

        db = sqlite3.connect(tmp_path / "keyed" / "audit.sqlite")
        row = db.execute(
            "SELECT status_code, n_rows FROM requests WHERE request_id = 'denied-1'"
        ).fetchone()
        assert row == (401, 0)


def test_admin_endpoints_are_closed_until_an_admin_key_is_set(
    served: dict[str, Any], tmp_path: Path
) -> None:
    """The public demo runs with no keys at all: scoring is open, labels and audit are not."""
    payload = _payload(served["rows"].iloc[0])
    label = {"transaction_id": 1, "is_fraud": True, "event_dt": 1, "observed_dt": 2}
    outcome = {"outcomes": [label]}
    with _fresh_client(served, tmp_path, admin_key=None) as c:
        h = c.get("/health").json()
        assert h["auth"] == "open" and h["admin"] == "disabled"
        assert c.post("/predict", json=payload).status_code == 200
        for method, path, kwargs in (
            ("post", "/outcomes", {"json": outcome}),
            ("get", "/audit/recent", {}),
            ("get", "/audit/recent", {"headers": {"X-API-Key": "anything"}}),
        ):
            r = getattr(c, method)(path, **kwargs)
            assert r.status_code == 403 and "FRAUD_ADMIN_API_KEY" in r.json()["detail"], path
            assert r.headers["X-Request-ID"]  # refused calls are still recorded
        assert c.get("/audit/recent?transaction_id=1").status_code == 403
    (tmp_path / "admin").mkdir()
    with _fresh_client(served, tmp_path / "admin", admin_key="adm1n") as c:
        assert c.get("/health").json()["admin"] == "api_key"
        assert c.post("/outcomes", json=outcome, headers={"X-API-Key": ""}).status_code == 401
        assert c.post("/outcomes", json=outcome, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/outcomes", json=outcome).status_code == 200  # default header = adm1n
        assert c.get("/audit/recent").status_code == 200
        assert c.post("/predict", json=payload, headers={"X-API-Key": ""}).status_code == 200


def test_request_time_budget_returns_504(served: dict[str, Any], tmp_path: Path) -> None:
    cfg = tmp_path / "serving.yaml"
    text = (
        Path(served["config"])
        .read_text()
        .replace(
            "model_path: models/m.joblib",
            f"model_path: {Path(served['config']).parents[1] / 'models' / 'm.joblib'}",
        )
    )
    cfg.write_text(text + "audit_db: null\nrequest_timeout_s: 0.001\n")
    rows: pd.DataFrame = served["rows"].head(50)
    csv = rows.drop(columns=[schema.TARGET_COL, schema.HAS_IDENTITY_COL]).to_csv(index=False)
    with TestClient(create_app(cfg)) as c:
        r = c.post("/predict/csv", files={"file": ("q.csv", csv, "text/csv")})
        assert r.status_code == 504 and "exceeded" in r.json()["detail"]
        assert c.get("/health").status_code == 200  # unguarded paths are not bounded


def test_every_guarded_call_logs_one_json_line(
    client: TestClient, served: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    import json
    import logging

    payload = _payload(served["rows"].iloc[0])
    with caplog.at_level(logging.INFO, logger="fraud.serve.requests"):
        client.post("/predict", json=payload, headers={"X-Request-ID": "log-1"})
        client.post("/explain", json=payload, headers={"X-Request-ID": "log-2"})
        client.post("/predict", json={"bad": 1}, headers={"X-Request-ID": "log-3"})
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "fraud.serve.requests"]
    by_id = {line["request_id"]: line for line in lines}
    assert by_id["log-1"]["path"] == "/predict" and by_id["log-1"]["status"] == 200
    assert by_id["log-1"]["rows"] == 1 and by_id["log-1"]["latency_ms"] > 0
    assert by_id["log-2"]["path"] == "/explain" and by_id["log-2"]["rows"] == 1
    assert by_id["log-3"]["status"] == 422 and by_id["log-3"]["rows"] == 0
    assert set(by_id["log-1"]) == {
        "ts", "request_id", "method", "path", "status", "latency_ms", "rows", "client"
    }  # fmt: skip


# --- champion fetched at startup (hosts whose image must not contain the weights) --------


def test_champion_is_fetched_from_a_private_store_at_startup(
    served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With FRAUD_CHAMPION_URL set and no local champion, the service downloads the
    artifact, golden and manifest with the bearer token, then runs the parity check."""
    import shutil
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from fraud.serve.app import CHAMPION_FILES, fetch_champion

    # a private store: the fixture model laid out as a champion directory
    store = tmp_path / "store"
    store.mkdir()
    src = Path(served["config"]).parents[1] / "models"
    shutil.copyfile(src / "m.joblib", store / "model.joblib")
    shutil.copyfile(src / "m_frozen_sample.json", store / "model_frozen_sample.json")
    shutil.copyfile(src / "m_frozen_expected.json", store / "model_frozen_expected.json")
    import hashlib
    import json

    sha = hashlib.sha256((store / "model.joblib").read_bytes()).hexdigest()
    (store / "model_manifest.json").write_text(
        json.dumps(
            {
                "run_name": "fixture", "model_version": "fixture+sigmoid", "artifact_sha256": sha,
                "bands": {"review": 0.05, "block": 0.5}, "model_info": {"experiment": "fetched"},
            }
        )
    )  # fmt: skip
    seen: list[tuple[str, str | None]] = []

    class Store(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen.append((self.path, self.headers.get("Authorization")))
            if self.headers.get("Authorization") != "Bearer tok":
                self.send_response(401)
                self.end_headers()
                return
            target = store / self.path.strip("/").split("/")[-1]
            if not target.exists():
                self.send_response(404)
                self.end_headers()
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Store)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # the production shape: the release tag carries the artifact digest, which is the
    # anchor fetch_champion verifies against before model.joblib is ever deserialized.
    base = f"http://127.0.0.1:{server.server_port}/releases/download/champion-{sha[:12]}"
    try:
        # the module function: fetches what is missing, tolerates a missing reference
        target = tmp_path / "champion" / "model.joblib"
        fetched = fetch_champion(target, base, "tok", sha[:12])
        assert set(fetched) == set(CHAMPION_FILES) - {"model_monitor_reference.json"}
        assert all(a == "Bearer tok" for _, a in seen)
        assert fetch_champion(target, base, "tok", sha[:12]) == []  # idempotent
        with pytest.raises(RuntimeError, match="HTTP 401"):
            fetch_champion(tmp_path / "other" / "model.joblib", base, "wrong", sha[:12])
        # the service: an empty champion directory + the env → a serving, parity-checked app
        root = tmp_path / "svc"
        (root / "configs").mkdir(parents=True)
        cfg = root / "configs" / "serving.yaml"
        # A fetched manifest supplies reporting fields but not bands, so the config must
        # already carry the promoted champion's bands or startup fails (docs/security.md).
        cfg.write_text(
            "model_path: models/champion/model.joblib\naudit_db: null\n"
            "model_version: unused\nbands: {review: 0.05, block: 0.5}\n"
        )
        monkeypatch.setenv("FRAUD_CHAMPION_URL", base)
        monkeypatch.setenv("FRAUD_CHAMPION_TOKEN", "tok")
        with TestClient(create_app(cfg)) as c:
            h = c.get("/health").json()
            assert h["parity_rows"] == 10 and h["model_version"].startswith("fixture+sigmoid@")
            # reporting fields still come from the fetched manifest ...
            assert c.get("/model-info").json()["experiment"] == "fetched"
            # ... while the bands served are the config's, which must agree with it
            assert c.get("/model-info").json()["bands"]["block"] == 0.5
        assert (root / "models" / "champion" / "model.joblib").exists()
    finally:
        server.shutdown()


def test_a_substituted_artifact_is_refused_before_it_is_deserialized(tmp_path: Path) -> None:
    """model.joblib is a pickle: loading it executes it. The digest pinned outside the
    store must be checked on the downloaded bytes, so a substituted artifact never
    reaches joblib.load. Proven with a pickle that writes a file when it is executed."""
    import hashlib
    import pickle
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from fraud.serve.app import expected_champion_digest, fetch_champion

    canary = tmp_path / "pwned"

    class Payload:
        def __reduce__(self) -> Any:
            return (Path.write_text, (canary, "code execution"))

    hostile = pickle.dumps(Payload())
    honest_digest = hashlib.sha256(b"the artifact that was promoted").hexdigest()

    # positive control: deserializing really does run the payload, so the assertion
    # below that the canary is absent is evidence and not a vacuous pass.
    pickle.loads(hostile)
    assert canary.read_text() == "code execution"
    canary.unlink()

    class Store(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = hostile if self.path.endswith("model.joblib") else b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Store)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = (
            f"http://127.0.0.1:{server.server_port}/releases/download/champion-{honest_digest[:12]}"
        )
        # the tag pins the digest; the store serves something else
        assert expected_champion_digest(base, None) == honest_digest[:12]
        target = tmp_path / "champion" / "model.joblib"
        with pytest.raises(RuntimeError, match="is not the pinned"):
            fetch_champion(target, base, None, honest_digest[:12])
        assert not canary.exists(), "the hostile pickle was executed"
        assert not target.exists(), "the rejected artifact was left on disk"
    finally:
        server.shutdown()


def test_an_unpinned_champion_url_is_refused(tmp_path: Path) -> None:
    """No champion-<sha12> tag and no FRAUD_CHAMPION_SHA256 means no anchor outside the
    store, so there is nothing to verify against: fail closed rather than trust it."""
    from fraud.serve.app import expected_champion_digest

    with pytest.raises(RuntimeError, match="unpinned champion"):
        expected_champion_digest("https://example.invalid/resolve/main", None)
    with pytest.raises(RuntimeError, match="FRAUD_CHAMPION_SHA256 must be"):
        expected_champion_digest("https://example.invalid/resolve/main", "not-hex")
    # an explicit pin is an anchor even when the URL carries none, and wins over the tag
    assert expected_champion_digest("https://example.invalid/x", "AABBCCDD") == "aabbccdd"
    tagged = "https://e.invalid/champion-0123456789ab"
    assert expected_champion_digest(tagged, None) == "0123456789ab"


def test_health_reports_the_commit_the_build_came_from(
    served: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release workflow pins the Render deploy to the tagged commit and then checks
    the live service reports it. The served version cannot answer that: every commit
    between one release bump and the next carries the same version."""
    from fraud.serve.app import build_commit

    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.delenv("FRAUD_BUILD_COMMIT", raising=False)
    assert build_commit() is None  # a host that says nothing claims nothing
    monkeypatch.setenv("FRAUD_BUILD_COMMIT", "  ")
    assert build_commit() is None  # blank is not a commit
    monkeypatch.setenv("FRAUD_BUILD_COMMIT", "deadbeefcafe")
    assert build_commit() == "deadbeefcafe"
    monkeypatch.setenv("RENDER_GIT_COMMIT", "6dd8b796742f0befdb4e9b6644616fa2040b2523")
    assert build_commit() == "6dd8b796742f0befdb4e9b6644616fa2040b2523"  # the host wins

    with TestClient(create_app(Path(served["config"]))) as c:
        body = c.get("/health").json()
    assert body["build_commit"] == "6dd8b796742f0befdb4e9b6644616fa2040b2523"


def test_a_fetched_manifest_cannot_set_the_policy_bands(
    served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """model.joblib is digest-pinned to its release tag; the manifest beside it is not.
    A manifest that keeps artifact_sha256 correct and changes `bands` would otherwise
    set the block and review thresholds from the network (docs/security.md)."""
    import hashlib
    import json
    import shutil
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    src = Path(served["config"]).parents[1] / "models"
    store = tmp_path / "store"
    store.mkdir()
    shutil.copyfile(src / "m.joblib", store / "model.joblib")
    shutil.copyfile(src / "m_frozen_sample.json", store / "model_frozen_sample.json")
    shutil.copyfile(src / "m_frozen_expected.json", store / "model_frozen_expected.json")
    sha = hashlib.sha256((store / "model.joblib").read_bytes()).hexdigest()
    honest_bands = {"review": 0.05, "block": 0.5}
    # the artifact is genuine and its digest is correct; only the policy is tampered with
    (store / "model_manifest.json").write_text(
        json.dumps(
            {
                "run_name": "fixture", "model_version": "fixture+sigmoid",
                "artifact_sha256": sha, "bands": {"review": 0.9, "block": 0.99},
                "model_info": {"experiment": "fetched"},
            }
        )
    )  # fmt: skip

    class Store(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            target = store / self.path.strip("/").split("/")[-1]
            if not target.exists():
                self.send_response(404)
                self.end_headers()
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Store)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        root = tmp_path / "svc"
        (root / "configs").mkdir(parents=True)
        cfg = root / "configs" / "serving.yaml"
        cfg.write_text(
            "model_path: models/champion/model.joblib\naudit_db: null\n"
            f"model_version: unused\nbands: {json.dumps(honest_bands)}\n"
        )
        monkeypatch.setenv(
            "FRAUD_CHAMPION_URL",
            f"http://127.0.0.1:{server.server_port}/releases/download/champion-{sha[:12]}",
        )
        # the tampered policy is refused rather than served, and named in the error
        # (the artifact is loaded in the lifespan, so the refusal surfaces on startup)
        with pytest.raises(RuntimeError, match="do not match"), TestClient(create_app(cfg)):
            pass
    finally:
        server.shutdown()


def test_model_info_reports_the_served_artifact_digest(served: dict[str, Any]) -> None:
    """The deployment contract: a release asserts that the artifact the service has
    loaded is the champion it was cut for (scripts/deploy_check.py). Consistency between
    /health, /model-info and a scored batch cannot show that — a stale champion is
    consistently the old model — so the digest itself is reported."""
    import hashlib

    artifact = Path(served["config"]).parents[1] / "models" / "m.joblib"
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    with TestClient(create_app(Path(served["config"]))) as c:
        info = c.get("/model-info").json()
        health = c.get("/health").json()

    assert info["artifact_sha256"] == expected, "reported digest is not the served artifact"
    # model_version carries the first 12 of the same digest, so the two cannot disagree
    assert health["model_version"].rpartition("@")[2] == expected[:12]
    assert info["version"] == health["model_version"]


def test_single_predictions_do_not_spend_the_review_budget(
    client: TestClient, served: dict[str, Any]
) -> None:
    """A /predict in the review band used to be audited as action='review', and
    reviewed_transactions() charges the day's rank budget with every such row, whatever
    endpoint wrote it. One policy's output quietly consumed the other's capacity. A
    single score decides nothing about review, so it must not record one."""
    rows = served["rows"]
    medium = next(
        (
            payload
            for i in range(len(rows))
            if (payload := _payload(rows.iloc[i]))
            and client.post("/predict", json=payload).json()["risk_level"] == "medium"
        ),
        None,
    )
    assert medium is not None, "fixture exercises no review band; the test would be vacuous"

    predict_rows = [
        e for e in client.get("/audit/recent?limit=200").json() if e["endpoint"] == "/predict"
    ]
    assert predict_rows, "single predictions were not audited at all"
    # the transaction sits in the review band and is still not recorded as a review
    charged = [e for e in predict_rows if e["action"] == "review"]
    assert not charged, f"{len(charged)} single scores charged the day's review budget"
    banded = [e for e in predict_rows if e["transaction_id"] == medium["TransactionID"]]
    assert banded and {e["action"] for e in banded} == {"approve"}
    assert {e["risk_level"] for e in banded} == {"medium"}  # the band is still reported


def _reference_for(served: dict[str, Any], fixture_raw_dir: Path) -> Path:
    """Freeze a monitoring reference next to the served fixture artifact."""
    from fraud.monitor.reference import build_reference, reference_path, save_reference

    artifact = Path(served["config"]).parents[1] / "models" / "m.joblib"
    parts = split(load_train(fixture_raw_dir), load_split_config(ROOT / "configs" / "split.yaml"))
    ref = build_reference(served["model"], parts["train"], parts["validation"], 0.42, 5, "abc")
    save_reference(ref, artifact)
    assert reference_path(artifact).exists()
    return artifact


def test_monitor_endpoint_matches_the_offline_report_over_the_same_trail(
    served: dict[str, Any], fixture_raw_dir: Path, tmp_path: Path
) -> None:
    """GET /audit/monitor is the offline monitor run where the audit trail lives (ADR 0011).

    Same `build_report`, same database, same window — so the scheduled job is reading
    the report `scripts/monitor.py` would have produced, not a second implementation.
    """
    from fraud.monitor.reference import load_reference, reference_path
    from fraud.monitor.report import build_report
    from fraud.serve.audit import AuditLog

    artifact = Path(served["config"]).parents[1] / "models" / "m.joblib"
    reference_path(artifact).unlink(missing_ok=True)  # the precondition, whatever ran before
    with _fresh_client(served, tmp_path) as c:
        missing = c.get("/audit/monitor")
        assert missing.status_code == 404
        assert "scripts/monitor_reference.py" in missing.json()["detail"]

        _reference_for(served, fixture_raw_dir)
        rows = [_payload(row) for _, row in served["rows"].head(30).iterrows()]
        assert c.post("/predict/batch", json={"transactions": rows}).status_code == 200

        report = c.get("/audit/monitor").json()
        assert report["status"] in ("ok", "warn", "alert")
        assert report["predictions"]["rows"] == 30 and report["data"]["rows"] == 30
        # the scoring call only: the monitor's own /audit calls are not the traffic it judges
        assert report["api"]["requests"] == 1 and report["api"]["error_rate"] == 0.0
        assert set(report["api"]["by_endpoint"]) == {"/predict/batch"}
        assert isinstance(report["flags"], list)

        audit = AuditLog(tmp_path / "audit.sqlite")
        offline = build_report(
            audit.frame("requests"),
            audit.frame("prediction_events"),
            audit.frame("input_features"),
            load_reference(artifact),
            None,
            (None, None),
            None,
            0,
        )
        assert offline["predictions"] == report["predictions"]
        assert offline["data"] == report["data"]
        assert offline["status"] == report["status"] and offline["flags"] == report["flags"]

        later = c.get("/audit/monitor", params={"since": "2099-01-01T00:00:00+00:00"}).json()
        assert later["predictions"] == {"rows": 0} and later["status"] == "ok"


def test_monitor_endpoint_refuses_a_window_wider_than_it_can_afford(
    served: dict[str, Any], fixture_raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The free host is 0.1 CPU: a window is counted before it is loaded into pandas."""
    _reference_for(served, fixture_raw_dir)
    cfg = tmp_path / "serving.yaml"
    text = (
        Path(served["config"])
        .read_text()
        .replace(
            "model_path: models/m.joblib",
            f"model_path: {Path(served['config']).parents[1] / 'models' / 'm.joblib'}",
        )
    )
    cfg.write_text(f"{text}audit_db: {tmp_path / 'capped.sqlite'}\nmonitor_max_rows: 5\n")
    monkeypatch.setenv("FRAUD_ADMIN_API_KEY", ADMIN_KEY)
    with TestClient(create_app(cfg), headers={"X-API-Key": ADMIN_KEY}) as c:
        rows = [_payload(row) for _, row in served["rows"].head(8).iterrows()]
        assert c.post("/predict/batch", json={"transactions": rows}).status_code == 200
        refused = c.get("/audit/monitor")
        assert refused.status_code == 413 and "shorter window" in refused.json()["detail"]


def test_monitor_endpoint_is_admin_only(served: dict[str, Any], tmp_path: Path) -> None:
    with _fresh_client(served, tmp_path, admin_key=None) as c:
        assert c.get("/audit/monitor").status_code == 403  # no admin key set: disabled outright
    (tmp_path / "keyed").mkdir()
    with _fresh_client(served, tmp_path / "keyed", admin_key="adm1n") as c:
        assert c.get("/audit/monitor", headers={"X-API-Key": "wrong"}).status_code == 401
