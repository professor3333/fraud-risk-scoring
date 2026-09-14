"""Evaluation metrics (ADR 0003). Every call returns the full G5 set, never one number."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
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
