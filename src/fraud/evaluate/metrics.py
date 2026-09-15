"""Evaluation metrics (ADR 0003). Every call returns the full G5 set, never one number."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def compute_metrics(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if s.min() < 0 or s.max() > 1:
        raise ValueError(f"scores must be probabilities in [0, 1]; got [{s.min()}, {s.max()}]")
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_curve, r_curve, _ = precision_recall_curve(y, s)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "recall_at_precision_0.90": _recall_at_precision(p_curve, r_curve, 0.90),
        "precision_at_recall_0.50": _precision_at_recall(p_curve, r_curve, 0.50),
        "positive_rate": float(y.mean()),
        "n": float(len(y)),
    }


def top_k_per_day(
    y_true: ArrayLike, y_score: ArrayLike, day: ArrayLike, k: int
) -> dict[str, float]:
    """Precision and recall when only the k highest-scored transactions per day are reviewed.

    The operational question for a capacity-limited review team: which transactions
    should analysts spend their k daily reviews on? Ties are broken by score order.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "s": np.asarray(y_score), "d": day})
    frame = frame.sort_values(["d", "s"], ascending=[True, False])
    frame["rank"] = frame.groupby("d").cumcount()
    reviewed = frame[frame["rank"] < k]
    tp = int(reviewed["y"].sum())
    n_pos = int(frame["y"].sum())
    return {
        f"precision_at_{k}_per_day": tp / len(reviewed) if len(reviewed) else 0.0,
        f"recall_at_{k}_per_day": tp / n_pos if n_pos else 0.0,
        f"reviewed_per_day_{k}": float(len(reviewed) / frame["d"].nunique()),
    }


def _recall_at_precision(p: np.ndarray, r: np.ndarray, min_precision: float) -> float:
    ok = p >= min_precision
    return float(r[ok].max()) if ok.any() else 0.0


def _precision_at_recall(p: np.ndarray, r: np.ndarray, min_recall: float) -> float:
    ok = r >= min_recall
    return float(p[ok].max()) if ok.any() else 0.0


def plot_pr_curve(y_true: ArrayLike, y_score: ArrayLike, title: str) -> Any:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    p, r, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(r, p, color="#2a78d6", lw=2)
    ax.axhline(y.mean(), color="#eb6834", lw=1, ls="--", label=f"prior = {y.mean():.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}  AP = {average_precision_score(y, s):.4f}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: ArrayLike, y_score: ArrayLike, title: str) -> Any:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#2a78d6", lw=2)
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1, ls="--", label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}  ROC-AUC = {roc_auc_score(y, s):.4f}")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig
