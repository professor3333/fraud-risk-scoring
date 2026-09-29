"""Threshold analysis and a three-action review policy (approve / review / block).

The model gives a calibrated probability; the business needs an action. This
module tabulates operating points, sizes a review band to an analyst budget,
and prices the resulting policy with ADR 0006's costs plus a review cost.
All selection happens on the validation window.
"""

from __future__ import annotations

import math
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


class NoBlockThresholdError(ValueError):
    """No threshold with enough flagged rows reaches the block precision bar."""


def block_threshold(points: pd.DataFrame, min_precision: float, min_flagged: int = 1) -> float:
    """Lowest threshold whose precision (and every higher threshold's) meets ``min_precision``.

    Scanning from the top keeps the rule monotone: once precision dips below the
    bar, nothing lower is allowed to block. Thresholds flagging fewer than
    ``min_flagged`` rows are skipped: a threshold above every score flags nothing,
    and its 0/0 precision (reported as 0) is no evidence either way. It used to end
    the scan at once, so scores [0.8, 0.1] with labels [1, 0] chose 0.9 over 0.5.
    Raises :class:`NoBlockThresholdError` when no supported threshold meets the bar,
    rather than falling back to the top of the grid as though one had.
    """
    ordered = points.sort_values("threshold", ascending=False)
    ordered = ordered[ordered["n_flagged"] >= min_flagged]
    chosen: float | None = None
    for t, p in zip(ordered["threshold"], ordered["precision"], strict=True):
        if p < min_precision:
            break
        chosen = float(t)
    if chosen is None:
        raise NoBlockThresholdError(
            f"no threshold flagging at least {min_flagged} row(s) reaches block precision "
            f"{min_precision}"
        )
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


def apply_rank_policy(
    y_score: ArrayLike, block_threshold: float, review_budget: int
) -> tuple[np.ndarray, float | None]:
    """Block by threshold; review the ``review_budget`` highest remaining scores; approve the rest.

    This is the deployment rule recommended in docs/review_policy.md: a fixed
    probability threshold for blocking, a fixed *count* for reviewing, so the
    review volume holds when the score distribution drifts. Returns the action
    per row and the lowest probability that was reviewed (None if nothing was).
    """
    s = np.asarray(y_score, dtype=float)
    action = np.where(s >= block_threshold, "block", "approve").astype(object)
    candidates = np.flatnonzero(s < block_threshold)
    budget = max(int(review_budget), 0)
    if budget and len(candidates):
        order = candidates[np.argsort(-s[candidates], kind="stable")][:budget]
        action[order] = "review"
        return action, float(s[order].min())
    return action, None


def apply_daily_rank_policy(
    y_score: ArrayLike,
    day: ArrayLike,
    block_threshold: float,
    capacity_by_day: dict[int, int],
    held: ArrayLike | None = None,
) -> tuple[np.ndarray, float | None]:
    """The rank policy applied within each transaction day, each with its own capacity.

    Serving form of :func:`apply_rank_policy`: a request may span days and a day
    may have spent part of its budget on earlier requests, so the capacity is
    supplied per day. ``held`` marks rows already in review from an earlier
    request (ADR 0013): they stay in review unless they now clear the block
    threshold, they take their day's capacity first, and the remaining rows are
    ranked into what is left. Returns the action per row and the lowest reviewed score.
    """
    s = np.asarray(y_score, dtype=float)
    d = np.asarray(day)
    h = np.zeros(len(s), dtype=bool) if held is None else np.asarray(held, dtype=bool)
    if h.shape != s.shape:
        raise ValueError(f"held has shape {h.shape}, scores {s.shape}")
    action = np.where(s >= block_threshold, "block", "approve").astype(object)
    keep = h & (s < block_threshold)
    action[keep] = "review"
    for value in np.unique(d):
        rows = np.flatnonzero((d == value) & ~h)
        left = capacity_by_day.get(int(value), 0) - int((keep & (d == value)).sum())
        day_actions, _ = apply_rank_policy(s[rows], block_threshold, max(left, 0))
        action[rows] = day_actions
    reviewed = s[action == "review"]
    return action, float(reviewed.min()) if len(reviewed) else None


def released_capacity(budget_per_day: int, transaction_dt: ArrayLike) -> dict[int, int]:
    """Reviews each transaction day has released by its latest transaction among these rows.

    The serving rule (docs/review_policy.md): the budget is released through the day,
    ``ceil(budget * fraction of the day elapsed)``, so a morning request cannot take the
    evening's queue. A whole-day upload gets the full budget.
    """
    dt = np.asarray(transaction_dt, dtype=np.int64)
    days = dt // SECONDS_PER_DAY
    out: dict[int, int] = {}
    for day in np.unique(days):
        latest = int(dt[days == day].max())
        elapsed = (latest - int(day) * SECONDS_PER_DAY + 1) / SECONDS_PER_DAY
        out[int(day)] = int(math.ceil(max(int(budget_per_day), 0) * elapsed))
    return out


def replay_served_policy(
    y_score: ArrayLike,
    transaction_dt: ArrayLike,
    block_threshold: float,
    budget_per_day: int,
    request_seconds: int | None = None,
) -> np.ndarray:
    """The actions the service would return if these rows arrived as requests.

    The one implementation of the served rank policy used offline: promotion, the
    monitoring reference and the replay report all call it, and serving calls the same
    two pieces (`released_capacity`, `apply_daily_rank_policy`) per request with its
    audit trail in place of the ``spent`` dictionary kept here.

    ``request_seconds=None`` sends each transaction day as one request, as a daily
    upload does: then it is exactly "block by threshold, review the day's top N".
    A number sends consecutive windows of that many seconds, in time order, as an
    online queue would: each request is ranked only against itself and the capacity
    the day has released so far, so a late high score can find the day's slots taken
    by earlier, lower ones. The gap between the two is what arrival order costs.
    """
    s = np.asarray(y_score, dtype=float)
    dt = np.asarray(transaction_dt, dtype=np.int64)
    if s.shape != dt.shape:
        raise ValueError(f"scores {s.shape} and times {dt.shape} differ")
    days = dt // SECONDS_PER_DAY
    window = days if request_seconds is None else dt // int(request_seconds)
    action = np.empty(len(s), dtype=object)
    spent: dict[int, int] = {}
    for w in np.unique(window):  # ascending: requests arrive in time order
        rows = np.flatnonzero(window == w)
        released = released_capacity(budget_per_day, dt[rows])
        capacity = {d: max(r - spent.get(d, 0), 0) for d, r in released.items()}
        got, _ = apply_daily_rank_policy(s[rows], days[rows], block_threshold, capacity)
        action[rows] = got
        for d in np.unique(days[rows]):
            spent[int(d)] = spent.get(int(d), 0) + int((got[days[rows] == d] == "review").sum())
    return action


def policy_outcome(
    y_true: ArrayLike,
    action: ArrayLike,
    amount: ArrayLike,
    n_days: int,
    costs: CostModel,
    review: ReviewCost,
    review_catch_rate: float = 1.0,
) -> dict[str, float]:
    """What a set of actions costs and catches, however they were decided.

    ``review_catch_rate`` is the share of reviewed fraud an analyst actually stops; the
    rest is approved in the end and costs a missed fraud. 1.0 is the assumption the
    cost model was built on (configs/policy.yaml); lower values are its sensitivity.
    """
    y = np.asarray(y_true, dtype=bool)
    a = np.asarray(amount, dtype=float)
    act = np.asarray(action)
    c = float(review_catch_rate)
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"review_catch_rate must be in [0, 1], got {c}")
    n_pos = int(y.sum())
    blk, rev, app = act == "block", act == "review", act == "approve"
    # costs: block = decline costs on legit; review = handling (+ delay on legit), and a
    # reviewed fraud is caught with probability c; approve = missed fraud
    cost = (
        costs.false_positive.of(a[blk & ~y]).sum()
        + (review.fixed + review.legit_amount_coef * a[rev & ~y]).sum()
        + review.fixed * (rev & y).sum()
        + (1 - c) * costs.false_negative.of(a[rev & y]).sum()
        + costs.false_negative.of(a[app & y]).sum()
    )
    caught = (blk & y).sum() + c * (rev & y).sum()
    return {
        "blocked_per_day": blk.sum() / n_days,
        "block_precision": (blk & y).sum() / blk.sum() if blk.sum() else 0.0,
        "reviewed_per_day": rev.sum() / n_days,
        "review_precision": (rev & y).sum() / rev.sum() if rev.sum() else 0.0,
        "fraud_blocked_per_day": (blk & y).sum() / n_days,
        "fraud_reviewed_per_day": (rev & y).sum() / n_days,
        "fraud_approved_per_day": (app & y).sum() / n_days,
        "recall_block": (blk & y).sum() / n_pos if n_pos else 0.0,
        "recall_block_plus_review": caught / n_pos if n_pos else 0.0,
        "legit_declined_per_day": (blk & ~y).sum() / n_days,
        "fraud_amount_caught_share": float((a[blk & y].sum() + c * a[rev & y].sum()) / a[y].sum())
        if y.any()
        else 0.0,
        "total_cost": float(cost),
    }


def evaluate_policy(
    y_true: ArrayLike,
    y_score: ArrayLike,
    amount: ArrayLike,
    day: ArrayLike,
    policy: Policy,
    costs: CostModel,
    review: ReviewCost,
) -> dict[str, float]:
    """The fixed-band policy (``policy="threshold"``) priced by :func:`policy_outcome`."""
    n_days = len(np.unique(np.asarray(day)))
    return {
        "block_threshold": policy.block_threshold,
        "review_threshold": policy.review_threshold,
        "budget_per_day": float(policy.budget_per_day),
        **policy_outcome(y_true, apply_policy(y_score, policy), amount, n_days, costs, review),
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
