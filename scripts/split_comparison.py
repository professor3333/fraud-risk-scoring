"""Educational experiment: random stratified split vs the frozen temporal split (E010).

Pools the temporal train and validation windows only (the test window is never
touched), draws a stratified random split of the same proportions, trains the
same model config on it, and reports both validation scores side by side.
The random number is NOT used for any decision; it exists to show what a
random split would have claimed. See ADR 0002 for the justification (G3).

Example:
    uv run python scripts/split_comparison.py --model configs/model/xgboost_v2_tuned.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud.data import schema
from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.features.columns import load_feature_spec
from fraud.train.run import fit_and_evaluate, git_commit, load_train_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "split_comparison")
    args = parser.parse_args()

    cfg = load_train_config(args.model)
    spec = load_feature_spec(cfg.features)
    temporal = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    pool = pd.concat([temporal["train"], temporal["validation"]])
    val_fraction = len(temporal["validation"]) / len(pool)

    # Random stratified split of the SAME pool, same sizes; the test window is not in the pool.
    rand_train, rand_val = train_test_split(
        pool,
        test_size=val_fraction,
        stratify=pool[schema.TARGET_COL],
        random_state=cfg.seed,
        shuffle=True,  # deliberate: this is the control, see ADR 0002
    )
    random_parts = {"train": rand_train.sort_index(), "validation": rand_val.sort_index()}

    results: dict[str, dict[str, float]] = {}
    for name, parts in (("temporal", temporal), ("random_stratified", random_parts)):
        _, train_m, val_m = fit_and_evaluate(parts, spec, cfg)
        results[name] = {
            "val_pr_auc": val_m["pr_auc"],
            "val_roc_auc": val_m["roc_auc"],
            "val_recall_at_precision_0.90": val_m["recall_at_precision_0.90"],
            "train_pr_auc": train_m["pr_auc"],
            "gap": train_m["pr_auc"] - val_m["pr_auc"],
            "n_train": len(parts["train"]),
            "n_val": len(parts["validation"]),
            "val_positive_rate": val_m["positive_rate"],
        }
        print(
            f"{name:18s} val PR-AUC {val_m['pr_auc']:.4f}  ROC {val_m['roc_auc']:.4f}", flush=True
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{cfg.run_name}.json").write_text(json.dumps(results, indent=2))
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fraud-methodology")
    with mlflow.start_run(run_name=f"{cfg.run_name}_random_vs_temporal"):
        mlflow.log_params(
            {"git_commit": git_commit(), "model_config": args.model.name, "seed": cfg.seed}
        )
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
