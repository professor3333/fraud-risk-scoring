"""Monitoring tests: drift arithmetic, the reference, the report, the request/input tables,
and the delayed-label feedback loop (outcomes, cohorts, the early signal)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud.monitor.drift import binned_shares, categorical_shares, level, psi, quantile_edges
from fraud.monitor.reference import build_reference
from fraud.monitor.report import build_report, render_markdown
from fraud.serve.audit import AuditLog


def test_psi_is_zero_for_identical_and_grows_with_shift() -> None:
    ref = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert psi(ref, ref) == pytest.approx(0.0)
    small = psi(ref, {"a": 0.52, "b": 0.29, "c": 0.19})
    big = psi(ref, {"a": 0.2, "b": 0.3, "c": 0.5})
    assert 0 < small < 0.01 < big
    assert level(small) == "ok" and level(0.15) == "warn" and level(big) == "alert"
    # a level absent on one side is handled, not a crash
    assert psi(ref, {"a": 1.0}) > 0


def test_shares_and_bins() -> None:
    s = pd.Series(["W", "W", "C", None])
    assert categorical_shares(s) == {"W": 0.5, "C": 0.25, "<missing>": 0.25}
    edges = quantile_edges(pd.Series([1.0, 2.0, 3.0, 4.0]), n_bins=2)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    b = binned_shares(pd.Series([0.5, 1.5, 2.5, 3.5, 10.0]), edges)
    assert sum(b.values()) == pytest.approx(1.0)


def test_reference_and_report_on_fixture(fixture_raw_dir: Path, tmp_path: Path) -> None:
    from fraud.data.load import load_train
    from fraud.data.split import load_split_config, split
    from fraud.features.columns import load_feature_spec
    from fraud.pipeline.build import build_pipeline
    from fraud.pipeline.calibrated import CalibratedModel

    root = Path(__file__).resolve().parents[1]
    parts = split(load_train(fixture_raw_dir), load_split_config(root / "configs" / "split.yaml"))
    spec = load_feature_spec(root / "configs" / "features" / "v2_freq.yaml")
    pipe = build_pipeline(spec, {"type": "xgboost", "params": {"n_estimators": 20}}, seed=0)
    pipe.fit(parts["train"], parts["train"][spec.target])
    held = parts["validation"]
    model = CalibratedModel(pipe, method="sigmoid").fit_calibrator(
        pipe.predict_proba(held)[:, 1], held[spec.target]
    )
    ref = build_reference(model, parts["train"], parts["validation"], 0.42, 5, "abc")
    assert set(ref["features"]) >= {"product", "card4", "has_identity", "amount", "hour"}
    assert sum(ref["actions"].values()) == pytest.approx(1.0)
    assert 0 <= ref["performance"]["pr_auc"] <= 1

    # simulate stored traffic: the test window scored, with inputs and requests
    from fraud.serve.audit import PredictionEvent, utc_now
    from fraud.serve.frames import monitored_fields

    test = parts["test"]
    p = model.predict_proba(test)[:, 1]
    audit = AuditLog(tmp_path / "audit.sqlite")
    now = utc_now()
    audit.record(
        [
            PredictionEvent("r1", "/predict/csv", now, int(t), "m@1", float(pp), "low",
                            "block" if pp >= 0.42 else "approve", "rank", 0.42, None, 5, None,
                            len(test), 12.5)
            for t, pp in zip(test["TransactionID"], p, strict=True)
        ]
    )  # fmt: skip
    audit.record_inputs(
        [{"request_id": "r1", "scored_at": now, **f} for f in monitored_fields(test)]
    )
    audit.record_request("r1", "/predict/csv", now, 200, 12.5, len(test))
    audit.record_request("r2", "/predict", now, 422, 1.0, 0)
    labels = test[["TransactionID", "isFraud"]].rename(
        columns={"TransactionID": "transaction_id", "isFraud": "is_fraud"}
    )
    report = build_report(
        audit.frame("requests"), audit.frame("prediction_events"), audit.frame("input_features"),
        ref, labels,
    )  # fmt: skip
    assert report["api"]["requests"] == 2 and report["api"]["errors"] == 1
    assert report["predictions"]["rows"] == len(test)
    assert set(report["data"]["features"]) == set(ref["features"])
    assert report["model"]["eventual"]["n_labelled"] == len(test)
    assert report["status"] in ("ok", "warn", "alert")
    md = render_markdown(report)
    assert "## API" in md and "## Data" in md and "eventual" in md
    # without labels the model section says so
    no_labels = build_report(audit.frame("requests"), audit.frame("prediction_events"),
                             audit.frame("input_features"), ref)  # fmt: skip
    assert no_labels["model"]["eventual"] is None

    # --- the delayed-label loop on the same traffic ------------------------------------
    from fraud.monitor.feedback import (
        SECONDS_PER_DAY,
        FeedbackConfig,
        arrived_by,
        simulate_arrivals,
    )

    cfg = FeedbackConfig(maturity_days=30, report_lag_median_days=5, report_lag_sigma=0.5, seed=1)
    # re-record the events with the transaction clock (the app fills it from the frame)
    audit2 = AuditLog(tmp_path / "audit2.sqlite")
    audit2.record(
        [
            PredictionEvent("r1", "/predict/csv", now, int(t), "m@1", float(pp), "low",
                            "block" if pp >= 0.42 else "approve", "rank", 0.42, None, 5, None,
                            len(test), 12.5, int(dt))
            for t, pp, dt in zip(test["TransactionID"], p, test["TransactionDT"], strict=True)
        ]
    )  # fmt: skip
    arrivals = simulate_arrivals(test[["TransactionID", "TransactionDT", "isFraud"]], cfg)
    first_day = int(test["TransactionDT"].min() // SECONDS_PER_DAY)
    # a few days in: only reports have arrived, no cohort has closed
    early_clock = (first_day + 10) * SECONDS_PER_DAY
    known = arrived_by(arrivals, early_clock).assign(recorded_at=now, source="test")
    audit2.record_outcomes(known.to_dict("records"))
    audit2.record_feed_run(early_clock, len(known), "test")
    rep = build_report(
        audit2.frame("requests"), audit2.frame("prediction_events"), audit2.frame("input_features"),
        ref, audit2.frame("outcomes"), clock=early_clock, maturity_days=cfg.maturity_days,
    )  # fmt: skip
    lab = rep["model"]["labels"]
    assert lab["closed_cohort"] == 0 and lab["overdue"] == 0
    assert lab["matured"] + lab["pending"] == len(test)
    assert "pr_auc" not in rep["model"]["eventual"]  # nothing final to evaluate yet
    if lab["matured"]:
        assert lab["positive_rate_arrived"] == 1.0  # only fraud has been reported
        assert rep["model"]["early"]["fraud_reported_so_far"] == lab["matured"]
    md = render_markdown(rep)
    assert "too few" in md and "label feed gap" not in md
    # after the window: every cohort closed, eventual metrics equal the complete-label ones
    late_clock = int(test["TransactionDT"].max()) + cfg.maturity_seconds + 1
    known = arrived_by(arrivals, late_clock).assign(recorded_at=now, source="test")
    audit2.record_outcomes(known.to_dict("records"))
    rep = build_report(
        audit2.frame("requests"), audit2.frame("prediction_events"), audit2.frame("input_features"),
        ref, audit2.frame("outcomes"), clock=late_clock, maturity_days=cfg.maturity_days,
    )  # fmt: skip
    lab = rep["model"]["labels"]
    assert lab["matured"] == len(test) == lab["closed_cohort"] and lab["pending"] == 0
    assert rep["model"]["eventual"]["pr_auc"] == pytest.approx(
        report["model"]["eventual"]["pr_auc"]
    )
    assert lab["positive_rate_closed"] == pytest.approx(lab["positive_rate_arrived"])
    # a feed that stops delivering shows up as overdue rows and an alert
    stale = build_report(
        audit2.frame("requests"), audit2.frame("prediction_events"), audit2.frame("input_features"),
        ref, audit2.frame("outcomes").iloc[:1], clock=late_clock, maturity_days=cfg.maturity_days,
    )  # fmt: skip
    assert stale["model"]["labels"]["overdue"] == len(test) - 1
    assert stale["status"] == "alert" and any("label feed gap" in f for f in stale["flags"])


def test_simulated_arrivals_respect_the_window_and_the_seed() -> None:
    from fraud.monitor.feedback import (
        SECONDS_PER_DAY,
        FeedbackConfig,
        arrived_by,
        attach_outcomes,
        early_section,
        label_section,
        mature_rows,
        simulate_arrivals,
    )

    cfg = FeedbackConfig(maturity_days=30, report_lag_median_days=7, report_lag_sigma=1.0, seed=3)
    n = 2000
    rng = np.random.default_rng(0)
    labels = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": rng.integers(0, 60 * SECONDS_PER_DAY, n),
            "isFraud": (rng.random(n) < 0.2).astype(int),
        }
    )
    a = simulate_arrivals(labels, cfg)
    lag = (a["observed_dt"] - a["event_dt"]) / SECONDS_PER_DAY
    pos, neg = a["is_fraud"] == 1, a["is_fraud"] == 0
    assert (lag[neg] == cfg.maturity_days).all()  # a negative is confirmed when the window closes
    assert (lag[pos] <= cfg.maturity_days).all() and (lag[pos] >= 0).all()
    assert lag[pos].median() < cfg.maturity_days / 2  # reports arrive well before the window
    # deterministic, and independent of which subset is passed
    again = simulate_arrivals(labels.sample(frac=1, random_state=9), cfg)
    assert again.equals(a)
    sub = simulate_arrivals(labels.iloc[::3], cfg)
    assert (
        sub.merge(a, on="transaction_id", suffixes=("", "_full"))["observed_dt"].equals(
            sub.merge(a, on="transaction_id", suffixes=("", "_full"))["observed_dt_full"]
        )
        is False
    )  # a subset is a different draw sequence: the feed always simulates the full frame

    # the positive-early bias: arrived labels overstate the fraud rate until cohorts close
    events = pd.DataFrame(
        {
            "transaction_id": labels["TransactionID"],
            "transaction_dt": labels["TransactionDT"],
            "fraud_probability": rng.random(n),
            "action": rng.choice(["block", "review", "approve"], n),
            "block_threshold": 0.5,
        }
    )
    clock = 46 * SECONDS_PER_DAY - 1  # end of day 45, as the feed script sets it
    attached = attach_outcomes(events, arrived_by(a, clock), clock, cfg.maturity_days)
    sec = label_section(attached, clock, cfg.maturity_days)
    assert sec["matured"] + sec["pending"] + sec["overdue"] == n and sec["overdue"] == 0
    assert sec["closed_cohort"] == int((labels["TransactionDT"] // SECONDS_PER_DAY <= 15).sum())
    true_rate = labels["isFraud"].mean()
    assert sec["positive_rate_arrived"] > true_rate + 0.1 > sec["positive_rate_closed"] - 0.1
    early = early_section(attached)
    assert 0 < early["fraud_reported_so_far"] < early["open_cohort"]
    assert 0 <= early["early_recall_block"] <= early["early_recall_block_plus_review"] <= 1
    # training data is the closed cohort, whatever has been reported since
    assert len(mature_rows(labels, 45, cfg.maturity_days)) == sec["closed_cohort"]
    # events without a transaction time are neither pending nor overdue
    untimed = attach_outcomes(
        events.assign(transaction_dt=np.nan).iloc[:5], a.iloc[:0], clock, cfg.maturity_days
    )
    assert (untimed["label_status"] == "unknown_time").all()
