"""Ablation and importance for a model config (§8 / Stage 5).

1. Gain importance from the saved pipeline (per feature, aggregated by group).
2. Group permutation importance on the validation window (no refit).
3. Ablation: refit the config with one feature group removed at a time and
   report the validation PR-AUC delta. Each refit is logged to MLflow.

Example:
    uv run python scripts/ablation.py --model-config configs/model/xgboost_v2_tuned.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.importance import (
    GROUP_PATTERNS,
    gain_importance,
    group_permutation_importance,
    spec_without_group,
)
from fraud.features.columns import load_feature_spec
from fraud.train.run import fit_and_evaluate, git_commit, load_train_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "ablation")
    parser.add_argument("--skip-refit", action="store_true", help="importance only, no ablation")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()

    cfg = load_train_config(args.model_config)
    spec = load_feature_spec(cfg.features)
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(args.models_dir / f"{cfg.run_name}.joblib")
    gain = gain_importance(pipe, spec)
    gain.to_csv(args.out_dir / f"{cfg.run_name}_gain.csv", index=False)
    gain_by_group = gain.groupby("group")["gain_share"].sum().sort_values(ascending=False)
    print("gain share by group:\n" + gain_by_group.round(4).to_string())
    print("\ntop 25 features by gain:\n" + gain.head(25)[["feature", "gain_share"]].to_string())

    perm = group_permutation_importance(pipe, parts["validation"], spec.target, spec, cfg.seed)
    perm.to_csv(args.out_dir / f"{cfg.run_name}_permutation.csv", index=False)
    print(f"\ngroup permutation importance (base PR-AUC {perm.attrs['base_pr_auc']:.4f}):")
    print(perm.round(4).to_string(index=False))

    if args.skip_refit:
        return

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fraud-ablation")
    _, _, full = fit_and_evaluate(parts, spec, cfg)
    base = full["pr_auc"]
    rows = []
    with mlflow.start_run(run_name=f"{cfg.run_name}_ablation"):
        mlflow.log_params({"git_commit": git_commit(), "base_run_name": cfg.run_name})
        mlflow.log_metric("base_val_pr_auc", base)
        for group in GROUP_PATTERNS:
            reduced = spec_without_group(spec, group)
            if len(reduced.all_inputs) == len(spec.all_inputs):
                continue
            _, train_m, val_m = fit_and_evaluate(parts, reduced, cfg)
            delta = val_m["pr_auc"] - base
            mlflow.log_metric(f"without_{group}_val_pr_auc", val_m["pr_auc"])
            rows.append(
                {
                    "removed_group": group,
                    "n_removed": len(spec.all_inputs) - len(reduced.all_inputs),
                    "val_pr_auc": val_m["pr_auc"],
                    "delta_vs_full": delta,
                    "train_val_gap": train_m["pr_auc"] - val_m["pr_auc"],
                }
            )
            print(
                f"without {group:13s} val PR-AUC {val_m['pr_auc']:.4f} ({delta:+.4f})", flush=True
            )
        (args.out_dir / f"{cfg.run_name}_ablation.json").write_text(
            json.dumps({"base_val_pr_auc": base, "ablation": rows}, indent=2)
        )
        mlflow.log_artifacts(str(args.out_dir), "ablation")


if __name__ == "__main__":
    main()
