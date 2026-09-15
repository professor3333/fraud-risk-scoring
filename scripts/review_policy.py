"""Threshold analysis and the three-action review policy on the validation window.

Writes reports/policy/: operating points at every threshold, the review-budget
curve, and one policy row per analyst budget (block threshold from the
precision bar, review band sized to the budget, costs under ADR 0006 + review
costs). The test window is scored only for the final policy table.

Example:
    uv run python scripts/review_policy.py --run-name xgb_f5_interactions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fraud.data.load import load_train  # noqa: E402
from fraud.data.split import load_split_config, split  # noqa: E402
from fraud.evaluate.policy import (  # noqa: E402
    Policy,
    budget_curve,
    evaluate_policy,
    load_policy_config,
    operating_points,
    policy_table,
)
from fraud.evaluate.threshold import load_threshold_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--policy-config", type=Path, default=ROOT / "configs" / "policy.yaml")
    parser.add_argument(
        "--threshold-config", type=Path, default=ROOT / "configs" / "threshold.yaml"
    )
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "policy")
    parser.add_argument("--with-test", action="store_true", help="also score the test window")
    args = parser.parse_args()

    pcfg = load_policy_config(args.policy_config)
    tcfg = load_threshold_config(args.threshold_config)
    model = joblib.load(args.models_dir / f"{args.run_name}_calibrated.joblib")
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def columns(frame: pd.DataFrame) -> tuple:
        return (
            frame["isFraud"].to_numpy(),
            model.predict_proba(frame)[:, 1],
            frame["TransactionAmt"].to_numpy(),
            frame["TransactionDT"].to_numpy() // 86_400,
        )

    y, p, amt, day = columns(parts["validation"])
    n_days = len(set(day))
    points = operating_points(y, p, n_days, tcfg.thresholds)
    points.to_csv(args.out_dir / f"{args.run_name}_operating_points.csv", index=False)
    curve = budget_curve(y, p, day, pcfg.review_budgets_per_day)
    curve.to_csv(args.out_dir / f"{args.run_name}_budget_curve_validation.csv", index=False)
    block_t, table = policy_table(y, p, amt, day, pcfg, tcfg.costs, tcfg.thresholds)
    table.to_csv(args.out_dir / f"{args.run_name}_policy_validation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].plot(points["threshold"], points["precision"], color="#2a78d6", lw=2, label="precision")
    axes[0].plot(points["threshold"], points["recall"], color="#eb6834", lw=2, label="recall")
    axes[0].plot(points["threshold"], points["f1"], color="#1baf7a", lw=1.5, label="F1")
    axes[0].axvline(block_t, color="#52514e", lw=1, ls="--")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Operating points (validation)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        curve["budget_per_day"],
        curve["recall_at_budget"],
        marker="o",
        color="#eb6834",
        lw=2,
        label="recall",
    )
    axes[1].plot(
        curve["budget_per_day"],
        curve["precision_at_budget"],
        marker="o",
        color="#2a78d6",
        lw=2,
        label="precision",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("reviews per day (budget)")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Review the top-k per day")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"{args.run_name}_policy.png", dpi=120)

    summary: dict[str, object] = {
        "block_threshold": block_t,
        "validation": table.to_dict("records"),
    }
    if args.with_test:
        yt, pt, amtt, dayt = columns(parts["test"])
        rows = []
        for _, r in table.iterrows():
            policy = Policy(
                float(r["block_threshold"]), float(r["review_threshold"]), int(r["budget_per_day"])
            )
            rows.append(evaluate_policy(yt, pt, amtt, dayt, policy, tcfg.costs, pcfg.review))
        test_table = pd.DataFrame(rows)
        test_table.to_csv(args.out_dir / f"{args.run_name}_policy_test.csv", index=False)
        summary["test"] = test_table.to_dict("records")
    (args.out_dir / f"{args.run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"block threshold (precision >= {pcfg.block_min_precision}): {block_t}")
    print(table.round(4).to_string(index=False))
    print(curve.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
