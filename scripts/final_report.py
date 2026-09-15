"""Final evaluation bundle on the frozen test window, plus an error-analysis table.

Writes reports/final/: metrics.json, precision_recall_curve.png, roc_curve.png,
confusion_matrix.png, threshold_analysis.csv, feature_importance.png,
error_analysis.csv. Uses the same deterministic predictions as the single test
evaluation (scripts/evaluate_test.py); nothing here feeds back into a decision.

Example:
    uv run python scripts/final_report.py --run-name xgb_f5_interactions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fraud.data import schema  # noqa: E402
from fraud.data.load import load_train  # noqa: E402
from fraud.data.split import load_split_config, split  # noqa: E402
from fraud.evaluate.calibration import calibration_metrics  # noqa: E402
from fraud.evaluate.importance import gain_importance  # noqa: E402
from fraud.evaluate.metrics import (  # noqa: E402
    compute_metrics,
    plot_pr_curve,
    plot_roc_curve,
    top_k_per_day,
)
from fraud.evaluate.policy import Policy, apply_policy, operating_points  # noqa: E402
from fraud.evaluate.threshold import cost_at, cost_curve, load_threshold_config  # noqa: E402
from fraud.features.columns import load_feature_spec  # noqa: E402
from fraud.train.run import load_train_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEGIT, FRAUD, GRID = "#2a78d6", "#eb6834", "#d9d8d3"

PROFILE_COLUMNS = [
    "TransactionAmt", "ProductCD", "card1", "card4", "card6", "addr1",
    "P_emaildomain", "R_emaildomain", "DeviceType",
    "C1", "C13", "D1", "D15", "M4", "id_31",
]  # fmt: skip


def plot_confusion(tp: int, fp: int, fn: int, tn: int, threshold: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cells = np.array([[tn, fp], [fn, tp]], dtype=float)
    ax.imshow(np.log1p(cells), cmap="Blues")
    for (i, j), v in np.ndenumerate(cells):
        ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=11,
                color="white" if v > cells.max() / 3 else "#0b0b0b")  # fmt: skip
    ax.set_xticks([0, 1], ["approve", "decline"])
    ax.set_yticks([0, 1], ["legit", "fraud"])
    ax.set_xlabel(f"decision at {threshold:.2f}")
    ax.set_ylabel("truth")
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_importance(gain: pd.DataFrame, out: Path) -> None:
    by_group = gain.groupby("group")["gain_share"].sum().sort_values()
    top = gain.head(20).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].barh(by_group.index, by_group.values * 100, color=LEGIT)
    axes[0].set_xlabel("share of split gain (%)")
    axes[0].set_title("Gain by feature group")
    axes[1].barh(
        top["feature"].str.replace("^(num|cat|freq)__", "", regex=True),
        top["gain_share"] * 100,
        color=LEGIT,
    )
    axes[1].set_xlabel("share of split gain (%)")
    axes[1].set_title("Top 20 features")
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument(
        "--threshold-config", type=Path, default=ROOT / "configs" / "threshold.yaml"
    )
    parser.add_argument("--block-threshold", type=float, default=0.42)
    parser.add_argument("--review-threshold", type=float, default=0.067)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "final")
    args = parser.parse_args()

    tcfg = load_threshold_config(args.threshold_config)
    model = joblib.load(args.models_dir / f"{args.run_name}_calibrated.joblib")
    model_cfg = (
        args.model_config
        or ROOT / "configs" / "model" / f"xgboost_{args.run_name.removeprefix('xgb_')}.yaml"
    )
    spec = load_feature_spec(load_train_config(model_cfg).features)
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    test = parts["test"]
    y = test[schema.TARGET_COL].to_numpy()
    amt = test["TransactionAmt"].to_numpy()
    day = test[schema.TIME_COL].to_numpy() // 86_400
    p = model.predict_proba(test)[:, 1]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    m = compute_metrics(y, p, tcfg.threshold)
    m.update({f"cal_{k}": v for k, v in calibration_metrics(y, p).items()})
    for k in (100, 200, 500):
        m.update(top_k_per_day(y, p, day, k))
    curve = cost_curve(y, p, amt, tcfg.costs, tcfg.thresholds)
    m["cost_at_threshold"] = cost_at(curve, tcfg.threshold)["total_cost"]
    m["cost_at_0.5"] = cost_at(curve, 0.5)["total_cost"]
    m["cost_approve_all"] = float(curve["fn_cost"].max())
    policy = Policy(args.block_threshold, args.review_threshold, 0)
    action = apply_policy(p, policy)
    n_days = len(np.unique(day))
    for a in ("block", "review", "approve"):
        mask = action == a
        m[f"policy_{a}_per_day"] = float(mask.sum() / n_days)
        m[f"policy_{a}_fraud_rate"] = float(y[mask].mean()) if mask.any() else 0.0
    m["policy_recall_block"] = float(((action == "block") & (y == 1)).sum() / y.sum())
    m["policy_recall_block_plus_review"] = float(((action != "approve") & (y == 1)).sum() / y.sum())
    (args.out_dir / "metrics.json").write_text(json.dumps(m, indent=2))

    # curves and matrix
    plot_pr_curve(y, p, f"{args.run_name} / test").savefig(
        args.out_dir / "precision_recall_curve.png", dpi=120
    )
    plot_roc_curve(y, p, f"{args.run_name} / test").savefig(args.out_dir / "roc_curve.png", dpi=120)
    plot_confusion(
        int(m["tp"]),
        int(m["fp"]),
        int(m["fn"]),
        int(m["tn"]),
        tcfg.threshold,
        args.out_dir / "confusion_matrix.png",
    )
    operating_points(y, p, n_days, tcfg.thresholds).to_csv(
        args.out_dir / "threshold_analysis.csv", index=False
    )

    # importance
    gain = gain_importance(model.pipeline, spec)
    gain.to_csv(args.out_dir / "feature_importance.csv", index=False)
    plot_importance(gain, args.out_dir / "feature_importance.png")

    # error analysis table: every test row with score, action, and error category
    err = test[[schema.ID_COL, schema.TIME_COL, schema.HAS_IDENTITY_COL, *PROFILE_COLUMNS]].copy()
    err.insert(1, "day", day)
    err["is_fraud"] = y
    err["probability"] = p
    err["action"] = action
    err["decision_at_threshold"] = np.where(p >= tcfg.threshold, "decline", "approve")
    err["category"] = np.select(
        [
            (p >= args.block_threshold) & (y == 0),
            (p >= args.block_threshold) & (y == 1),
            (p < args.review_threshold) & (y == 1),
            (p < args.review_threshold) & (y == 0),
        ],
        [
            "high_confidence_false_positive",
            "confident_true_fraud",
            "high_confidence_false_negative",
            "confident_true_legit",
        ],
        default="review_band",
    )
    err.sort_values("probability", ascending=False).to_csv(
        args.out_dir / "error_analysis.csv", index=False
    )
    print(
        json.dumps(
            {
                k: round(v, 4)
                for k, v in m.items()
                if k.startswith(("pr_auc", "roc_auc", "precision", "recall", "policy_"))
            },
            indent=2,
        )
    )
    print(err["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
