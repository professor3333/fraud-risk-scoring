"""Monthly retraining lifecycle, simulated offline (champion / challenger).

At each calendar cut-off T the labels are final only through M = T -
label_maturity_days (docs/feedback.md): a transaction younger than that has a
label only if it was reported, and training on those would be training on a
positive-enriched sample. So: train the challenger on days <= M - month_days,
calibrate it on out-of-fold folds inside its training data, score the latest
mature month (M - month_days + 1 .. M) with both challenger and incumbent, promote
the challenger if it beats the incumbent by the margin, re-select the block
threshold on that month, and freeze the promoted artifact with a golden for
parity. No decision reads labels past M. The month the artifact then serves
(T + 1 .. T + month_days) is scored after the fact as the eventual number the
monitor would report once those labels mature; it informs nothing in the cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from fraud.data import schema
from fraud.evaluate.metrics import compute_metrics
from fraud.evaluate.policy import block_threshold, operating_points
from fraud.features.columns import FeatureSpec, load_feature_spec
from fraud.pipeline.build import build_pipeline
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.parity import choose_sample, freeze
from fraud.train.run import load_train_config

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class RetrainConfig:
    model_config: Path
    month_days: int
    cutoffs: tuple[int, ...]
    calibration_folds: tuple[int, ...]
    promotion_margin: float
    block_min_precision: float
    label_maturity_days: int = 0  # 0: labels are final the day the transaction happens
    served_eval_through: int | None = None  # last day the served month may be scored on


def load_retrain_config(path: Path) -> RetrainConfig:
    raw = yaml.safe_load(path.read_text())
    served = raw.get("served_eval_through")
    return RetrainConfig(
        model_config=Path(raw["model_config"]),
        month_days=int(raw["month_days"]),
        cutoffs=tuple(int(c) for c in raw["cutoffs"]),
        calibration_folds=tuple(int(k) for k in raw["calibration_folds"]),
        promotion_margin=float(raw["promotion_margin"]),
        block_min_precision=float(raw["block_min_precision"]),
        label_maturity_days=int(raw.get("label_maturity_days", 0)),
        served_eval_through=None if served is None else int(served),
    )


@dataclass
class Incumbent:
    name: str
    model: CalibratedModel
    block_threshold: float
    trained_through: int


def _day(df: pd.DataFrame) -> pd.Series:
    return df[schema.TIME_COL] // SECONDS_PER_DAY


def fit_challenger(
    df: pd.DataFrame,
    spec: FeatureSpec,
    model_cfg: dict[str, Any],
    seed: int,
    train_end: int,
    cfg: RetrainConfig,
) -> CalibratedModel:
    """Fit on days <= train_end; calibrate on OOF folds strictly inside that range."""
    day = _day(df)
    train = df.loc[day <= train_end]
    pipe = build_pipeline(spec, model_cfg, seed)
    pipe.fit(train, train[spec.target])
    oof_scores: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []
    for k in cfg.calibration_folds:
        fold_end = train_end - k * cfg.month_days
        if fold_end < cfg.month_days:  # not enough history for this fold
            continue
        fit = df.loc[day <= fold_end]
        score = df.loc[(day > fold_end) & (day <= fold_end + cfg.month_days)]
        fold_pipe = build_pipeline(spec, model_cfg, seed)
        fold_pipe.fit(fit, fit[spec.target])
        oof_scores.append(fold_pipe.predict_proba(score)[:, 1])
        oof_y.append(score[spec.target].to_numpy())
    if not oof_scores:
        raise ValueError(f"no calibration fold fits before day {train_end}")
    return CalibratedModel(pipe, method="sigmoid").fit_calibrator(
        np.concatenate(oof_scores), np.concatenate(oof_y)
    )


def evaluate_on_month(model: CalibratedModel, month: pd.DataFrame, target: str) -> dict[str, float]:
    return compute_metrics(month[target], model.predict_proba(month)[:, 1], 0.5)


def select_block_threshold(
    model: CalibratedModel, month: pd.DataFrame, target: str, min_precision: float
) -> float:
    p = model.predict_proba(month)[:, 1]
    n_days = int(_day(month).nunique())
    points = operating_points(month[target], p, n_days, np.round(np.arange(0.01, 0.99, 0.005), 6))
    return block_threshold(points, min_precision)


def eventual_on_served_month(
    model: CalibratedModel,
    df: pd.DataFrame,
    cutoff: int,
    block_t: float,
    target: str,
    cfg: RetrainConfig,
) -> dict[str, Any]:
    """The month this artifact serves, scored once its labels would have matured.

    Descriptive only: the number the monitor reports label_maturity_days after the
    month ends. Skipped past served_eval_through so the reporting window is not
    consulted by accident (ADR 0002).
    """
    end = cutoff + cfg.month_days
    if cfg.served_eval_through is not None and end > cfg.served_eval_through:
        return {"served_month": None}
    day = _day(df)
    served = df.loc[(day > cutoff) & (day <= end)]
    if served.empty:
        return {"served_month": None}
    p = model.predict_proba(served)[:, 1]
    y = served[target].to_numpy(dtype=int)
    m = compute_metrics(y, p, block_t)
    blocked = p >= block_t
    return {
        "served_month": f"{cutoff + 1}-{end}",
        "served_positives": int(y.sum()),
        "served_pr_auc": m["pr_auc"],
        "served_block_precision": float(y[blocked].mean()) if blocked.any() else None,
        "served_recall_block": float(y[blocked].sum() / max(y.sum(), 1)),
        "staleness_days": end - (cutoff - cfg.label_maturity_days - cfg.month_days),
    }


def run_lifecycle(
    df: pd.DataFrame, cfg: RetrainConfig, out_dir: Path, seed: int | None = None
) -> pd.DataFrame:
    """Simulate every cut-off in order; returns one row per cycle."""
    tc = load_train_config(cfg.model_config)
    spec = load_feature_spec(tc.features)
    seed = tc.seed if seed is None else seed
    out_dir.mkdir(parents=True, exist_ok=True)
    incumbent: Incumbent | None = None
    rows: list[dict[str, Any]] = []
    day = _day(df)
    for t in cfg.cutoffs:
        mature_through = t - cfg.label_maturity_days
        train_end = mature_through - cfg.month_days
        if train_end < cfg.month_days:
            raise ValueError(
                f"cut-off {t}: labels are final only through day {mature_through} "
                f"({cfg.label_maturity_days}-day maturity); nothing to train on"
            )
        month = df.loc[(day > train_end) & (day <= mature_through)]
        if month.empty:
            raise ValueError(f"no rows in the validation month ending on day {mature_through}")
        challenger = fit_challenger(df, spec, tc.model, seed, train_end, cfg)
        ch = evaluate_on_month(challenger, month, spec.target)
        row: dict[str, Any] = {
            "cutoff": t,
            "mature_through": mature_through,
            "train_end": train_end,
            "month": f"{train_end + 1}-{mature_through}",
            "n_month": len(month),
            "positives": int(month[spec.target].sum()),
            "challenger_pr_auc": ch["pr_auc"],
            "challenger_roc_auc": ch["roc_auc"],
        }
        if incumbent is None:
            promoted, reason = True, "no incumbent (bootstrap)"
            row["incumbent"] = None
            row["incumbent_pr_auc"] = None
        else:
            inc = evaluate_on_month(incumbent.model, month, spec.target)
            row["incumbent"] = incumbent.name
            row["incumbent_pr_auc"] = inc["pr_auc"]
            delta = ch["pr_auc"] - inc["pr_auc"]
            row["delta"] = delta
            promoted = delta >= cfg.promotion_margin
            reason = (
                f"challenger +{delta:.4f} >= margin {cfg.promotion_margin}"
                if promoted
                else f"challenger {delta:+.4f} < margin {cfg.promotion_margin}; incumbent kept"
            )
        chosen = challenger if promoted else incumbent.model  # type: ignore[union-attr]
        name = f"{tc.run_name}_through_day{train_end}" if promoted else incumbent.name  # type: ignore[union-attr]
        block_t = select_block_threshold(chosen, month, spec.target, cfg.block_min_precision)
        chosen_metrics = evaluate_on_month(chosen, month, spec.target)
        # freeze the artifact that will serve the next month, with a golden from this month
        cycle_dir = out_dir / f"cutoff_{t}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        artifact = cycle_dir / f"{name}_calibrated.joblib"
        joblib.dump(chosen, artifact)
        freeze(chosen, artifact, choose_sample(month, n_per_group=25))
        (cycle_dir / "decision.json").write_text(
            json.dumps(
                {
                    "cutoff": t,
                    "promoted": promoted,
                    "reason": reason,
                    "serving": name,
                    "block_threshold": block_t,
                    "validation_month": row["month"],
                    "metrics_on_month": chosen_metrics,
                },
                indent=2,
            )
        )
        row.update(
            {
                "promoted": promoted,
                "reason": reason,
                "serving": name,
                "block_threshold": block_t,
                "serving_pr_auc": chosen_metrics["pr_auc"],
                "serving_recall_at_p90": chosen_metrics["recall_at_precision_0.90"],
                "artifact": str(artifact),
                **eventual_on_served_month(chosen, df, t, block_t, spec.target, cfg),
            }
        )
        rows.append(row)
        print(
            f"cutoff {t}: challenger {ch['pr_auc']:.4f}"
            + (f" vs incumbent {row['incumbent_pr_auc']:.4f}" if incumbent else "")
            + f" -> {'PROMOTE' if promoted else 'KEEP'} ({reason}); block threshold {block_t}",
            flush=True,
        )
        if promoted:
            incumbent = Incumbent(name, chosen, block_t, train_end)
        else:
            incumbent.block_threshold = block_t  # type: ignore[union-attr]
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "lifecycle.csv", index=False)
    return table
