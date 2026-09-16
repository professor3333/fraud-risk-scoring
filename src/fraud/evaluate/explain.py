"""Per-prediction explanation: which inputs moved this transaction's score, and how far.

XGBoost's ``pred_contribs`` gives exact TreeSHAP contributions for the fitted
booster — one number per model input per row, in log-odds, summing with the
bias to the raw margin. No extra dependency. Two things are added on top:

1. **Mapping back to the analyst's vocabulary.** The model's 491 inputs are
   post-``ColumnTransformer`` columns: one column per one-hot level, one per
   frequency table, one per raw numeric. Contributions are summed per *source*
   (``ProductCD``, not ``cat__ProductCD_C`` + ``cat__ProductCD_W`` + …) and per
   family (the ablation groups), and the row's own value is shown beside each.
2. **The calibration layer, stated.** Contributions explain the pipeline's raw
   score; the served probability is that score after the calibration map
   (ADR 0007). The explanation reports both and never pretends the pieces sum
   to the calibrated number.

Under ADR 0001 this answers "why does this transaction *rank* high", not "why is
it fraud": a routine purchase on a reported account carries the same signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from fraud.evaluate.importance import group_of
from fraud.pipeline.calibrated import CalibratedModel


@dataclass(frozen=True)
class Signal:
    feature: str  # the source column the analyst knows
    value: str  # the row's value, as text
    contribution: float  # log-odds on the raw score; > 0 raises the score
    family: str


@dataclass(frozen=True)
class Explanation:
    transaction_id: int
    fraud_probability: float  # served: calibrated
    raw_probability: float  # the pipeline's score before calibration
    raw_margin: float  # log-odds; bias + sum of all contributions
    bias: float  # the booster's baseline log-odds (the training prior, roughly)
    signals: list[Signal]  # top-k by |contribution|
    other_contribution: float  # everything not in the top-k, summed
    families: dict[str, float]  # contribution summed per feature family

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _sources(pipeline: Pipeline) -> tuple[list[str], list[str], list[str]]:
    """For each model input: its name, its source column and its family."""
    ct = pipeline.named_steps["features"]
    categorical = [c for name, _, c in ct.transformers_ if name == "cat"]
    cat_cols = list(categorical[0]) if categorical else []
    names, sources, families = [], [], []
    for name in ct.get_feature_names_out():
        prefix, raw = str(name).split("__", 1)
        if prefix == "freq":  # kept apart from the raw column: "card1 frequency" vs "card1"
            source, family = f"{raw.removeprefix('freq_')} frequency", "frequency"
        elif prefix == "cat":
            matches = [c for c in cat_cols if raw == c or raw.startswith(c + "_")]
            source = max(matches, key=len) if matches else raw
            family = group_of(source)
        else:
            source, family = raw, group_of(raw)
        names.append(str(name))
        sources.append(source)
        families.append(family)
    return names, sources, families


def contributions(pipeline: Pipeline, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """TreeSHAP contributions per model input (n_rows × n_inputs) and the bias per row."""
    transformed = pipeline[:-1].transform(frame)
    booster = pipeline.named_steps["model"].get_booster()
    raw = booster.predict(xgb.DMatrix(np.asarray(transformed, dtype=float)), pred_contribs=True)
    return raw[:, :-1], raw[:, -1]


def _display_value(frame: pd.DataFrame, source: str, transformed: np.ndarray, col: int) -> str:
    if source in frame.columns:
        v = frame[source].iloc[0]
        if pd.isna(v):
            return "missing"
        return f"{v:g}" if isinstance(v, float | np.floating) else str(v)
    x = float(transformed[0, col])  # frequency tables, derived interactions
    return "missing" if np.isnan(x) else f"{x:g}"


def explain(model: CalibratedModel, frame: pd.DataFrame, top_k: int = 8) -> Explanation:
    """Explain one row (a one-row frame in the production input shape)."""
    if len(frame) != 1:
        raise ValueError("explain one row at a time")
    pipeline = model.pipeline
    names, sources, families = _sources(pipeline)
    contrib, bias = contributions(pipeline, frame)
    c = contrib[0]
    margin = float(bias[0] + c.sum())
    raw_p = _sigmoid(margin)
    served = float(model.calibrate_scores(np.array([raw_p]))[0])
    transformed = np.asarray(pipeline[:-1].transform(frame), dtype=float)

    by_source: dict[str, float] = {}
    first_col: dict[str, int] = {}
    family_of: dict[str, str] = {}
    for i, (source, family) in enumerate(zip(sources, families, strict=True)):
        by_source[source] = by_source.get(source, 0.0) + float(c[i])
        first_col.setdefault(source, i)
        family_of[source] = family
    ordered = sorted(by_source.items(), key=lambda kv: -abs(kv[1]))
    top = ordered[:top_k]
    signals = [
        Signal(
            feature=source,
            value=_display_value(frame, source, transformed, first_col[source]),
            contribution=round(value, 4),
            family=family_of[source],
        )
        for source, value in top
    ]
    by_family: dict[str, float] = {}
    for source, value in by_source.items():
        by_family[family_of[source]] = by_family.get(family_of[source], 0.0) + value
    return Explanation(
        transaction_id=int(frame["TransactionID"].iloc[0]),
        fraud_probability=served,
        raw_probability=raw_p,
        raw_margin=margin,
        bias=float(bias[0]),
        signals=signals,
        other_contribution=round(sum(v for _, v in ordered[top_k:]), 4),
        families={k: round(v, 4) for k, v in sorted(by_family.items(), key=lambda kv: -abs(kv[1]))},
    )
