"""Monitoring tests: drift arithmetic, the reference, the report, and the request/input tables."""

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
    labels = test[["TransactionID", "isFraud"]].rename(columns={"TransactionID": "transaction_id"})
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
