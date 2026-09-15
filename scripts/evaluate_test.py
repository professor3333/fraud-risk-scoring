"""Evaluate a calibrated candidate on the final temporal reporting window.

Reports the full metric set at the configured threshold on both validation
(for the results table) and test, plus the cost model's totals. Every run is
logged to MLflow so that each consultation of the reporting window leaves a
record; ADR 0002 keeps the human-readable log. The window is never used to
select between candidates.

Example:
    uv run python scripts/evaluate_test.py --run-name xgb_v2_tuned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.calibration import calibration_metrics
from fraud.evaluate.metrics import compute_metrics, plot_pr_curve, top_k_per_day
from fraud.evaluate.threshold import cost_at, cost_curve, load_threshold_config
from fraud.train.run import git_commit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument(
        "--threshold-config", type=Path, default=ROOT / "configs" / "threshold.yaml"
    )
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "test")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()

    tcfg = load_threshold_config(args.threshold_config)
    model = joblib.load(args.models_dir / f"{args.run_name}_calibrated.joblib")
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, float]] = {}
    for window in ("validation", "test"):
        frame = parts[window]
        y = frame["isFraud"].to_numpy()
        amt = frame["TransactionAmt"].to_numpy()
        p = model.predict_proba(frame)[:, 1]
        m = compute_metrics(y, p, tcfg.threshold)
        day = frame["TransactionDT"].to_numpy() // 86_400
        for k in (100, 200, 500):
            m.update(top_k_per_day(y, p, day, k))
        m.update({f"cal_{k}": v for k, v in calibration_metrics(y, p).items()})
        curve = cost_curve(y, p, amt, tcfg.costs, tcfg.thresholds)
        m["cost_at_threshold"] = cost_at(curve, tcfg.threshold)["total_cost"]
        m["cost_approve_all"] = float(curve["fn_cost"].max())
        m["cost_at_0.5"] = cost_at(curve, 0.5)["total_cost"]
        report[window] = m
        fig = plot_pr_curve(y, p, f"{args.run_name} / {window}")
        fig.savefig(args.out_dir / f"{args.run_name}_pr_curve_{window}.png", dpi=120)

    (args.out_dir / f"{args.run_name}_report.json").write_text(json.dumps(report, indent=2))
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fraud-final")
    with mlflow.start_run(run_name=f"{args.run_name}_test_evaluation"):
        mlflow.log_params(
            {"git_commit": git_commit(), "run_name": args.run_name, "threshold": tcfg.threshold}
        )
        for window, m in report.items():
            mlflow.log_metrics({f"{window}_{k}": v for k, v in m.items()})
        mlflow.log_artifacts(str(args.out_dir), "test")

    keys = (
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "recall_at_precision_0.90",
        "precision_at_100_per_day",
        "recall_at_100_per_day",
        "precision_at_500_per_day",
        "recall_at_500_per_day",
        "cal_brier",
        "cal_ece",
        "cost_at_threshold",
        "cost_at_0.5",
        "cost_approve_all",
    )
    print(f"{'metric':28s} {'validation':>12s} {'test':>12s}")
    for k in keys:
        print(f"{k:28s} {report['validation'][k]:12.4f} {report['test'][k]:12.4f}")


if __name__ == "__main__":
    main()
