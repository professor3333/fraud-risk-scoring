"""Simulate the monthly retraining lifecycle offline (champion / challenger).

    new month arrives -> mature labels appended -> retrain preprocessing + model
    -> calibrate -> re-select block threshold -> evaluate challenger vs incumbent
    -> promote if the rule passes -> freeze the artifact (+ golden)

Development data (days <= 152) by default. --include-reporting-window adds a
cut-off at day 183, which scores the reporting window and is therefore logged as
a consultation in ADR 0002.

Example:
    uv run python scripts/retrain.py
    uv run python scripts/retrain.py --model-config configs/model/xgboost_f5_interactions.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import mlflow
import pandas as pd

from fraud.data.load import load_train
from fraud.train.lifecycle import load_retrain_config, run_lifecycle
from fraud.train.run import git_commit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "retrain.yaml")
    parser.add_argument("--model-config", type=Path, default=None, help="challenger recipe")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--label-maturity-days", type=int, default=None,
        help="labels are final only this many days after the transaction (docs/feedback.md)",
    )  # fmt: skip
    parser.add_argument("--include-reporting-window", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models" / "retrain")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "retrain")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()

    cfg = load_retrain_config(args.config)
    if args.model_config is not None:
        cfg = replace(cfg, model_config=args.model_config)
    if args.label_maturity_days is not None:
        cfg = replace(cfg, label_maturity_days=args.label_maturity_days)
    if args.include_reporting_window:
        cfg = replace(cfg, cutoffs=tuple(cfg.cutoffs) + (183,), served_eval_through=None)
        print("NOTE: cut-off 183 scores the reporting window; record it in ADR 0002's log.")
    df = load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fraud-retrain")
    with mlflow.start_run(run_name=f"lifecycle_{cfg.model_config.stem}"):
        mlflow.log_params(
            {
                "git_commit": git_commit(),
                "model_config": cfg.model_config.name,
                "cutoffs": ",".join(map(str, cfg.cutoffs)),
                "promotion_margin": cfg.promotion_margin,
                "block_min_precision": cfg.block_min_precision,
                "label_maturity_days": cfg.label_maturity_days,
            }
        )
        table = run_lifecycle(df, cfg, args.out_dir, args.seed)
        for _, r in table.iterrows():
            mlflow.log_metric(
                f"challenger_pr_auc_cutoff{int(r.cutoff)}", float(r.challenger_pr_auc)
            )
            mlflow.log_metric(f"serving_pr_auc_cutoff{int(r.cutoff)}", float(r.serving_pr_auc))
            if pd.notna(r.get("served_pr_auc")):
                mlflow.log_metric(f"served_pr_auc_cutoff{int(r.cutoff)}", float(r.served_pr_auc))
        args.report_dir.mkdir(parents=True, exist_ok=True)
        table.drop(columns=["artifact"]).to_csv(args.report_dir / "lifecycle.csv", index=False)
        mlflow.log_text(table.to_csv(index=False), "lifecycle.csv")
    pd.set_option("display.width", 220)
    cols = [
        "cutoff", "mature_through", "month", "positives", "incumbent_pr_auc", "challenger_pr_auc",
        "promoted", "serving", "block_threshold", "serving_pr_auc", "served_month", "served_pr_auc",
        "served_block_precision",
    ]  # fmt: skip
    print(table[[c for c in cols if c in table]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
