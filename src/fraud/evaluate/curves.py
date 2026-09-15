"""Learning curves for a fitted boosted-tree pipeline, computed after the fact.

Scores train and validation at increasing tree counts via ``iteration_range``,
so the validation window is never passed to ``fit`` — it is only *read* here,
which is what validation is for.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def learning_curve(
    pipe: Pipeline,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: str,
    step: int = 50,
) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    n_trees = int(model.get_booster().num_boosted_rounds())
    x_train = pipe[:-1].transform(train)
    x_val = pipe[:-1].transform(validation)
    rows: list[dict[str, float]] = []
    for k in list(range(step, n_trees, step)) + [n_trees]:
        p_tr = model.predict_proba(x_train, iteration_range=(0, k))[:, 1]
        p_va = model.predict_proba(x_val, iteration_range=(0, k))[:, 1]
        rows.append(
            {
                "trees": float(k),
                "train_pr_auc": float(average_precision_score(train[target], p_tr)),
                "val_pr_auc": float(average_precision_score(validation[target], p_va)),
            }
        )
    out = pd.DataFrame(rows)
    out["gap"] = out["train_pr_auc"] - out["val_pr_auc"]
    return out


def plot_learning_curve(curve: pd.DataFrame, title: str) -> Any:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(curve["trees"], curve["train_pr_auc"], color="#2a78d6", lw=2, label="train")
    ax.plot(curve["trees"], curve["val_pr_auc"], color="#eb6834", lw=2, label="validation")
    ax.set_xlabel("trees")
    ax.set_ylabel("PR-AUC")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
