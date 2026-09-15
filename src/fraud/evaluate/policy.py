"""Threshold analysis and a three-action review policy (approve / review / block).

The model gives a calibrated probability; the business needs an action. This
module tabulates operating points, sizes a review band to an analyst budget,
and prices the resulting policy with ADR 0006's costs plus a review cost.
All selection happens on the validation window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike

from fraud.evaluate.threshold import CostModel

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class ReviewCost:
    fixed: float
    legit_amount_coef: float


@dataclass(frozen=True)
class PolicyConfig:
    block_min_precision: float
    review_budgets_per_day: tuple[int, ...]
    review: ReviewCost


def load_policy_config(path: Path) -> PolicyConfig:
    raw = yaml.safe_load(path.read_text())
    return PolicyConfig(
        block_min_precision=float(raw["block_min_precision"]),
        review_budgets_per_day=tuple(int(b) for b in raw["review_budgets_per_day"]),
        review=ReviewCost(
            float(raw["costs"]["review_fixed"]), float(raw["costs"]["review_legit_amount_coef"])
        ),
    )


# --- operating points --------------------------------------------------------------


def operating_points(
    y_true: ArrayLike, y_score: ArrayLike, n_days: float, thresholds: ArrayLike
) -> pd.DataFrame:
    """Precision, recall, F1, the confusion counts and flagged volume at each threshold."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=float)
    n_pos = int(y.sum())
    rows = []
    for t in np.asarray(thresholds, dtype=float):
        flagged = s >= t
        tp = int((flagged & y).sum())
        fp = int((flagged & ~y).sum())
        fn = n_pos - tp
        tn = int(len(y) - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_pos if n_pos else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "n_flagged": tp + fp,
                "flagged_per_day": (tp + fp) / n_days,
                "fraud_caught_per_day": tp / n_days,
            }
        )
    return pd.DataFrame(rows)


def block_threshold(points: pd.DataFrame, min_precision: float) -> float:
    """Lowest threshold whose precision (and every higher threshold's) meets ``min_precision``.

    Scanning from the top keeps the rule monotone: once precision dips below the
    bar, nothing lower is allowed to block.
    """
    ordered = points.sort_values("threshold", ascending=False)
    chosen = float(ordered["threshold"].iloc[0])
    for t, p in zip(ordered["threshold"], ordered["precision"], strict=True):
        if p < min_precision:
            break
        chosen = float(t)
    return chosen


# --- review budget -----------------------------------------------------------------


def budget_curve(
    y_true: ArrayLike, y_score: ArrayLike, day: ArrayLike, budgets: tuple[int, ...]
) -> pd.DataFrame:
    """Pure ranking view: review the top-k per day, for several k."""
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "s": np.asarray(y_score), "d": day})
    frame = frame.sort_values(["d", "s"], ascending=[True, False])
    frame["rank"] = frame.groupby("d").cumcount()
    n_pos, n_days = int(frame["y"].sum()), frame["d"].nunique()
    rows = []
    for k in budgets:
        reviewed = frame[frame["rank"] < k]
        tp = int(reviewed["y"].sum())
        # the score of the last reviewed transaction, averaged over days: the implied cut
        implied = reviewed.groupby("d")["s"].min().mean()
        rows.append(
            {
                "budget_per_day": k,
                "precision_at_budget": tp / len(reviewed) if len(reviewed) else 0.0,
                "recall_at_budget": tp / n_pos if n_pos else 0.0,
                "fraud_caught_per_day": tp / n_days,
                "implied_threshold": float(implied),
            }
        )
    return pd.DataFrame(rows)


# --- three-action policy -----------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    block_threshold: float
    review_threshold: float
    budget_per_day: int


def size_review_band(
    y_score: ArrayLike, day: ArrayLike, block_t: float, budget_per_day: int
) -> float:
    """Highest ``review_threshold`` such that the band [review_t, block_t) holds at most
    ``budget_per_day`` transactions per day on average (validation sizing)."""
    s = np.asarray(y_score, dtype=float)
    d = np.asarray(day)
    n_days = len(np.unique(d))
    below_block = np.sort(s[s < block_t])[::-1]
    capacity = int(round(budget_per_day * n_days))
    if capacity <= 0 or len(below_block) == 0:
        return block_t
    if capacity >= len(below_block):
        return float(below_block[-1])
    return float(below_block[capacity - 1])


def apply_policy(y_score: ArrayLike, policy: Policy) -> np.ndarray:
    s = np.asarray(y_score, dtype=float)
    return np.where(
        s >= policy.block_threshold,
        "block",
        np.where(s >= policy.review_threshold, "review", "approve"),
    )


def evaluate_policy(
    y_true: ArrayLike,
    y_score: ArrayLike,
    amount: ArrayLike,
    day: ArrayLike,
    policy: Policy,
    costs: CostModel,
    review: ReviewCost,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=bool)
    a = np.asarray(amount, dtype=float)
    action = apply_policy(y_score, policy)
    n_days = len(np.unique(np.asarray(day)))
    n_pos = int(y.sum())
    blk, rev, app = action == "block", action == "review", action == "approve"
    # costs: block = decline costs on legit; review = handling (+ delay on legit), fraud caught;
    # approve = missed fraud
    cost = (
        costs.false_positive.of(a[blk & ~y]).sum()
        + (review.fixed + review.legit_amount_coef * a[rev & ~y]).sum()
        + review.fixed * (rev & y).sum()
        + costs.false_negative.of(a[app & y]).sum()
    )
    return {
        "block_threshold": policy.block_threshold,
        "review_threshold": policy.review_threshold,
        "budget_per_day": float(policy.budget_per_day),
        "blocked_per_day": blk.sum() / n_days,
        "block_precision": (blk & y).sum() / blk.sum() if blk.sum() else 0.0,
        "reviewed_per_day": rev.sum() / n_days,
        "review_precision": (rev & y).sum() / rev.sum() if rev.sum() else 0.0,
        "fraud_blocked_per_day": (blk & y).sum() / n_days,
        "fraud_reviewed_per_day": (rev & y).sum() / n_days,
        "fraud_approved_per_day": (app & y).sum() / n_days,
        "recall_block": (blk & y).sum() / n_pos,
        "recall_block_plus_review": ((blk | rev) & y).sum() / n_pos,
        "legit_declined_per_day": (blk & ~y).sum() / n_days,
        "total_cost": float(cost),
    }


def policy_table(
    y_true: ArrayLike,
    y_score: ArrayLike,
    amount: ArrayLike,
    day: ArrayLike,
    cfg: PolicyConfig,
    costs: CostModel,
    thresholds: ArrayLike,
) -> tuple[float, pd.DataFrame]:
    """Block threshold from the precision bar; one policy row per review budget."""
    n_days = len(np.unique(np.asarray(day)))
    points = operating_points(y_true, y_score, n_days, thresholds)
    block_t = block_threshold(points, cfg.block_min_precision)
    rows: list[dict[str, Any]] = []
    for budget in cfg.review_budgets_per_day:
        review_t = size_review_band(y_score, day, block_t, budget)
        policy = Policy(block_t, review_t, budget)
        rows.append(evaluate_policy(y_true, y_score, amount, day, policy, costs, cfg.review))
    return block_t, pd.DataFrame(rows)
