"""Rolling temporal backtests with forecast horizons (ADR 0008).

For each training cut-off T a candidate is fit on days <= T and scored on later
30-day windows that start after a gap of 0, 7, 14, 30 or 60 days. The result is a
horizon curve per candidate; selection uses the mean over gaps >= 30 days.
Everything here reads days <= max_day only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fraud.data import schema
from fraud.evaluate.metrics import compute_metrics
from fraud.features.columns import FeatureSpec
from fraud.pipeline.build import build_pipeline

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Horizon:
    train_end: int
    gap: int
    start: int
    end: int  # inclusive


@dataclass(frozen=True)
class BacktestConfig:
    experiment: str
    max_day: int
    window_days: int
    train_ends: tuple[int, ...]
    gaps: tuple[int, ...]
    robust_min_gap: int


def load_backtest_config(path: Path) -> BacktestConfig:
    raw = yaml.safe_load(path.read_text())
    return BacktestConfig(
        experiment=str(raw["experiment"]),
        max_day=int(raw["max_day"]),
        window_days=int(raw["window_days"]),
        train_ends=tuple(int(t) for t in raw["train_ends"]),
        gaps=tuple(int(g) for g in raw["gaps"]),
        robust_min_gap=int(raw["robust_min_gap"]),
    )


def horizons(cfg: BacktestConfig) -> list[Horizon]:
    """Every (train_end, gap) whose scored window ends by ``max_day``."""
    out: list[Horizon] = []
    for t in cfg.train_ends:
        for g in cfg.gaps:
            start, end = t + g + 1, t + g + cfg.window_days
            if end <= cfg.max_day:
                out.append(Horizon(train_end=t, gap=g, start=start, end=end))
    return out


def _day(df: pd.DataFrame) -> pd.Series:
    return df[schema.TIME_COL] // SECONDS_PER_DAY


def run_candidate(
    name: str,
    df: pd.DataFrame,
    spec: FeatureSpec,
    model_cfg: dict[str, Any],
    seed: int,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Fit once per training cut-off, score every horizon window; one row per window."""
    day = _day(df)
    if int(day.max()) > cfg.max_day:
        df = df.loc[day <= cfg.max_day]
        day = _day(df)
    rows: list[dict[str, Any]] = []
    for t in cfg.train_ends:
        train = df.loc[day <= t]
        pipe = build_pipeline(spec, model_cfg, seed)
        pipe.fit(train, train[spec.target])
        for h in (h for h in horizons(cfg) if h.train_end == t):
            window = df.loc[(day >= h.start) & (day <= h.end)]
            if window[schema.TIME_COL].min() <= train[schema.TIME_COL].max():
                raise ValueError(f"horizon {h} overlaps its training window")
            m = compute_metrics(window[spec.target], pipe.predict_proba(window)[:, 1], 0.5)
            rows.append(
                {
                    "candidate": name,
                    "train_end": t,
                    "gap": h.gap,
                    "window": f"{h.start}-{h.end}",
                    "n": len(window),
                    "positives": int(window[spec.target].sum()),
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                    "recall_at_p90": m["recall_at_precision_0.90"],
                }
            )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, robust_min_gap: int) -> pd.DataFrame:
    """Per candidate: adjacent mean, robust (gap >= min) mean, decay, and per-gap means."""
    out = []
    for name, g in results.groupby("candidate", sort=False):
        adjacent = g.loc[g["gap"] == 0, "pr_auc"].mean()
        robust = g.loc[g["gap"] >= robust_min_gap, "pr_auc"].mean()
        row: dict[str, Any] = {
            "candidate": name,
            "adjacent_pr_auc": float(adjacent),
            "robust_pr_auc": float(robust),
            "decay": float(adjacent - robust),
            "n_windows": int(len(g)),
        }
        for gap in sorted(set(g["gap"].astype(int))):
            row[f"gap_{gap}"] = float(g.loc[g["gap"] == gap, "pr_auc"].mean())
        out.append(row)
    return pd.DataFrame(out).sort_values("robust_pr_auc", ascending=False).reset_index(drop=True)
