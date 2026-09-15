"""Amount-aware cost curve and operating-threshold selection (ADR 0006)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class ErrorCost:
    fixed: float
    amount_coef: float

    def of(self, amount: np.ndarray) -> np.ndarray:
        return self.fixed + self.amount_coef * amount


@dataclass(frozen=True)
class CostModel:
    false_negative: ErrorCost
    false_positive: ErrorCost

    def scaled(self, fn_factor: float = 1.0, fp_factor: float = 1.0) -> CostModel:
        fn, fp = self.false_negative, self.false_positive
        return CostModel(
            ErrorCost(fn.fixed * fn_factor, fn.amount_coef * fn_factor),
            ErrorCost(fp.fixed * fp_factor, fp.amount_coef * fp_factor),
        )


@dataclass(frozen=True)
class ThresholdConfig:
    costs: CostModel
    thresholds: np.ndarray
    threshold: float


def _error_cost(raw: dict[str, Any]) -> ErrorCost:
    return ErrorCost(float(raw["fixed"]), float(raw["amount_coef"]))


def load_threshold_config(path: Path) -> ThresholdConfig:
    raw = yaml.safe_load(path.read_text())
    c = raw["costs"]
    sweep = raw["sweep"]
    grid = np.round(
        np.arange(float(sweep["start"]), float(sweep["stop"]) + 1e-12, float(sweep["step"])), 6
    )
    return ThresholdConfig(
        costs=CostModel(
            ErrorCost(
                float(c["false_negative"]["fixed"]), float(c["false_negative"]["amount_coef"])
            ),
            ErrorCost(
                float(c["false_positive"]["fixed"]), float(c["false_positive"]["amount_coef"])
            ),
        ),
        thresholds=grid,
        threshold=float(raw["threshold"]),
    )


def cost_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    amount: ArrayLike,
    costs: CostModel,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Total cost and confusion counts at each threshold. Costs use each row's own amount."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=float)
    a = np.asarray(amount, dtype=float)
    fn_cost = costs.false_negative.of(a)
    fp_cost = costs.false_positive.of(a)
    rows: list[dict[str, float]] = []
    for t in thresholds:
        flagged = s >= t
        tp = flagged & y
        fp = flagged & ~y
        fn = ~flagged & y
        n_tp, n_fp, n_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
        precision = n_tp / (n_tp + n_fp) if n_tp + n_fp else 0.0
        recall = n_tp / (n_tp + n_fn) if n_tp + n_fn else 0.0
        rows.append(
            {
                "threshold": float(t),
                "fn_cost": float(fn_cost[fn].sum()),
                "fp_cost": float(fp_cost[fp].sum()),
                "total_cost": float(fn_cost[fn].sum() + fp_cost[fp].sum()),
                "tp": n_tp,
                "fp": n_fp,
                "fn": n_fn,
                "flagged_rate": float(flagged.mean()),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    return pd.DataFrame(rows)


def select_threshold(curve: pd.DataFrame) -> float:
    i = int(curve["total_cost"].to_numpy().argmin())
    return float(curve["threshold"].to_numpy()[i])


def cost_at(curve: pd.DataFrame, threshold: float) -> dict[str, float]:
    i = int((curve["threshold"] - threshold).abs().argmin())
    keys = ("threshold", "total_cost", "precision", "recall", "f1")
    return {k: float(curve[k].to_numpy()[i]) for k in keys}


def sensitivity_table(
    y_true: ArrayLike,
    y_score: ArrayLike,
    amount: ArrayLike,
    costs: CostModel,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Re-select the threshold under ±50 % on each cost assumption."""
    rows: list[dict[str, Any]] = []
    for label, fn_f, fp_f in (
        ("base", 1.0, 1.0),
        ("FN cost -50 %", 0.5, 1.0),
        ("FN cost +50 %", 1.5, 1.0),
        ("FP cost -50 %", 1.0, 0.5),
        ("FP cost +50 %", 1.0, 1.5),
    ):
        curve = cost_curve(y_true, y_score, amount, costs.scaled(fn_f, fp_f), thresholds)
        best = cost_at(curve, select_threshold(curve))
        rows.append({"scenario": label, **best})
    return pd.DataFrame(rows)


def plot_cost_curve(curve: pd.DataFrame, chosen: float, title: str) -> Any:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(curve["threshold"], curve["total_cost"], color="#2a78d6", lw=2, label="total")
    axes[0].plot(
        curve["threshold"], curve["fn_cost"], color="#eb6834", lw=1.5, label="missed fraud"
    )
    axes[0].plot(
        curve["threshold"], curve["fp_cost"], color="#1baf7a", lw=1.5, label="false declines"
    )
    axes[0].axvline(chosen, color="#52514e", lw=1, ls="--")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("cost on the decision set")
    axes[0].set_title(title)
    axes[0].legend(frameon=False)
    axes[1].plot(curve["threshold"], curve["precision"], color="#2a78d6", lw=2, label="precision")
    axes[1].plot(curve["threshold"], curve["recall"], color="#eb6834", lw=2, label="recall")
    axes[1].axvline(chosen, color="#52514e", lw=1, ls="--")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return fig
