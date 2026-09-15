"""Calibration assessment: reliability table, Brier score, expected calibration error (ADR 0007)."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import brier_score_loss

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def reliability_table(y_true: ArrayLike, y_score: ArrayLike, n_bins: int = 15) -> pd.DataFrame:
    """Quantile bins of the score: mean predicted vs observed fraud rate per bin."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    rows: list[dict[str, float]] = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": float(b),
                "n": float(m.sum()),
                "score_mean": float(s[m].mean()),
                "observed_rate": float(y[m].mean()),
                "score_lo": float(edges[b]),
                "score_hi": float(edges[b + 1]),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(
    y_true: ArrayLike, y_score: ArrayLike, n_bins: int = 15
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    table = reliability_table(y, s, n_bins)
    ece = float(
        (table["n"] / table["n"].sum() * (table["score_mean"] - table["observed_rate"]).abs()).sum()
    )
    return {
        "brier": float(brier_score_loss(y, s)),
        "brier_prior": float(brier_score_loss(y, np.full_like(s, y.mean()))),
        "ece": ece,
        "mean_score": float(s.mean()),
        "positive_rate": float(y.mean()),
    }


def plot_reliability(tables: dict[str, pd.DataFrame], title: str) -> Any:
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1, ls="--", label="perfect")
    for (name, t), c in zip(tables.items(), colors, strict=False):
        ax.plot(t["score_mean"], t["observed_rate"], marker="o", ms=4, lw=1.5, color=c, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed fraud rate (bin)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
