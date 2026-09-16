"""Per-prediction explanation: contributions reproduce the raw score, sources partition the
model's inputs, the served probability is the calibrated one, and the endpoint agrees."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud.data import schema
from fraud.evaluate.explain import _sources, contributions, explain


def _payload(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col, val in row.items():
        if col in (schema.TARGET_COL, schema.HAS_IDENTITY_COL) or pd.isna(val):
            continue
        if col in (schema.ID_COL, schema.TIME_COL):
            out[col] = int(val)
        else:
            out[col] = val.item() if hasattr(val, "item") else val
    return out


def test_contributions_sum_to_the_raw_score(served: dict[str, Any]) -> None:
    model, rows = served["model"], served["rows"].head(25)
    contrib, bias = contributions(model.pipeline, rows)
    n_inputs = len(model.pipeline.named_steps["features"].get_feature_names_out())
    assert contrib.shape == (25, n_inputs)
    margin = bias + contrib.sum(axis=1)
    raw = model.raw_scores(rows)
    np.testing.assert_allclose(1 / (1 + np.exp(-margin)), raw, atol=1e-5)  # float32 booster
    # every model input maps to exactly one source and one family
    names, sources, families = _sources(model.pipeline)
    assert len(names) == len(sources) == len(families) == n_inputs
    assert "ProductCD" in sources and any(s.endswith(" frequency") for s in sources)
    assert set(families) <= {
        "amount", "product", "card", "addr", "dist", "email", "C", "D", "M", "V",
        "identity", "has_identity", "frequency", "other",
    }  # fmt: skip


def test_explanation_is_grouped_and_states_calibration(served: dict[str, Any]) -> None:
    model, rows = served["model"], served["rows"]
    p = model.predict_proba(rows)[:, 1]
    row = rows.iloc[[int(np.argmax(p))]]
    e = explain(model, row, top_k=5)
    assert e.transaction_id == int(row[schema.ID_COL].iloc[0])
    assert e.fraud_probability == pytest.approx(float(p.max()), abs=1e-6)  # the served number
    assert e.raw_probability == pytest.approx(float(model.raw_scores(row)[0]), abs=1e-5)
    assert e.raw_probability != e.fraud_probability  # calibration is a separate, stated step
    assert len(e.signals) == 5
    mags = [abs(s.contribution) for s in e.signals]
    assert mags == sorted(mags, reverse=True)
    # one-hot levels are summed per source: no signal is a post-transform column name
    assert not any("__" in s.feature for s in e.signals)
    assert len({s.feature for s in e.signals}) == 5
    # the pieces account for the whole margin
    shown = sum(s.contribution for s in e.signals)
    assert e.bias + shown + e.other_contribution == pytest.approx(e.raw_margin, abs=1e-3)
    assert sum(e.families.values()) == pytest.approx(e.raw_margin - e.bias, abs=1e-3)
    # the row's own value is shown next to each signal
    for s in e.signals:
        if s.feature in row.columns and not pd.isna(row[s.feature].iloc[0]):
            assert s.value not in ("", "missing")
    with pytest.raises(ValueError, match="one row"):
        explain(model, rows.head(2))


def test_explain_endpoint_matches_the_module_and_the_prediction(
    client: TestClient, served: dict[str, Any]
) -> None:
    row = served["rows"].iloc[0]
    payload = _payload(row)
    predicted = client.post("/predict", json=payload).json()
    r = client.post("/explain?top_k=6", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transaction_id"] == predicted["transaction_id"]
    assert body["fraud_probability"] == pytest.approx(predicted["fraud_probability"], abs=1e-9)
    assert body["model_version"] == predicted["model_version"]
    assert len(body["signals"]) == 6 and "calibration" in body["note"]
    direct = explain(served["model"], served["rows"].iloc[[0]], top_k=6)
    assert [s["feature"] for s in body["signals"]] == [s.feature for s in direct.signals]
    # validation is the prediction endpoint's
    assert client.post("/explain", json={**payload, "TransactionAmt": -1}).status_code == 422
    assert client.post("/explain", json={"TransactionID": 1}).status_code == 422
    # explaining decides nothing, so it is not in the audit trail
    before = client.get("/health").json()["audit_events"]
    client.post("/explain", json=payload)
    assert client.get("/health").json()["audit_events"] == before
