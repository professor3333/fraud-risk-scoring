"""The monitoring reference: what 'normal' looked like when the model was built.

Computed once per served artifact from the training population (features) and
the validation window (scores, actions, performance), and stored next to the
artifact so the monitor always compares live traffic with the population the
model and its policy were fitted on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud.data import schema
from fraud.evaluate.calibration import calibration_metrics
from fraud.evaluate.metrics import compute_metrics
from fraud.evaluate.policy import apply_rank_policy
from fraud.monitor.drift import (
    binned_shares,
    categorical_shares,
    fixed_width_edges,
    numeric_summary,
    quantile_edges,
)
from fraud.serve.frames import monitored_fields

CATEGORICAL = ("product", "card4", "card6", "device_type", "hour")
BINARY = ("has_identity", "addr1_missing", "p_email_present", "r_email_present")
NUMERIC = ("amount", "n_missing_transaction", "n_missing_identity")
COUNT_LIKE = ("n_missing_transaction", "n_missing_identity")  # discrete: equal-width bins
SCORE_EDGES = [-np.inf, 0.005, 0.01, 0.02, 0.04, 0.062, 0.1, 0.2, 0.42, 0.7, np.inf]


def reference_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.stem + "_monitor_reference.json")


def build_reference(
    model: Any,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    block_threshold: float,
    review_budget_per_day: int,
    artifact_sha256: str,
) -> dict[str, Any]:
    feats = pd.DataFrame(monitored_fields(train))
    ref: dict[str, Any] = {
        "artifact_sha256": artifact_sha256,
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "features": {},
        "numeric_edges": {},
    }
    for col in CATEGORICAL:
        ref["features"][col] = categorical_shares(feats[col])
    for col in BINARY:
        ref["features"][col] = {"rate": float(feats[col].mean())}
    for col in NUMERIC:
        edges = fixed_width_edges(feats[col]) if col in COUNT_LIKE else quantile_edges(feats[col])
        ref["numeric_edges"][col] = edges
        ref["features"][col] = {
            "bins": binned_shares(feats[col], edges),
            **numeric_summary(feats[col]),
        }

    p = np.asarray(model.predict_proba(validation)[:, 1], dtype=float)
    y = validation[schema.TARGET_COL].to_numpy()
    days = int((validation[schema.TIME_COL] // 86_400).nunique())
    actions, _ = apply_rank_policy(p, block_threshold, review_budget_per_day * days)
    ref["scores"] = {
        "edges": [float(e) for e in SCORE_EDGES],
        "bins": binned_shares(pd.Series(p), SCORE_EDGES),
        **numeric_summary(pd.Series(p)),
    }
    ref["actions"] = {a: float((actions == a).mean()) for a in ("block", "review", "approve")}
    ref["performance"] = {
        **{
            k: v
            for k, v in compute_metrics(y, p, block_threshold).items()
            if k in ("pr_auc", "roc_auc", "precision", "recall")
        },
        **{f"cal_{k}": v for k, v in calibration_metrics(y, p).items() if k in ("brier", "ece")},
        "block_precision": float(y[actions == "block"].mean())
        if (actions == "block").any()
        else 0.0,
        "positive_rate": float(y.mean()),
    }
    return ref


def save_reference(ref: dict[str, Any], artifact: Path) -> Path:
    path = reference_path(artifact)
    path.write_text(json.dumps(ref, indent=2))
    return path


def load_reference(artifact: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(reference_path(artifact).read_text())
    return loaded
