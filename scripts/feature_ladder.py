"""E018: cumulative feature-set ladder and family cuts, with the tuned parameters.

Ladder:  F0 → +F1 → +F2 → +F3 → +F4 → +F5 → +F6 → +F7 (each built on the previous).
Cuts on the shipped set: transaction only · transaction + identity · everything
except V · everything · everything except Vesta-engineered (C, D, M, V).

Every fit is a run in MLflow experiment `fraud-feature-sets`; validation only.

Example:
    uv run python scripts/feature_ladder.py --model configs/model/xgboost_f5_interactions.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.feature_sets import family_cuts, ladder
from fraud.features.columns import FeatureSpec, load_feature_spec
from fraud.features.history import ENTITY_DEFINITIONS, add_entity_history
from fraud.train.run import TrainConfig, fit_and_evaluate, git_commit, load_train_config

ROOT = Path(__file__).resolve().parents[1]


def evaluate(
    label: str, spec: FeatureSpec, df: pd.DataFrame, cfg: TrainConfig, split_path: Path
) -> dict[str, float | str | int]:
    frame = df
    if spec.history:
        frame = add_entity_history(df, ENTITY_DEFINITIONS[spec.history_entity])
    parts = split(frame, load_split_config(split_path))
    _, train_m, val_m = fit_and_evaluate(parts, spec, cfg)
    row: dict[str, float | str | int] = {
        "set": label,
        "n_inputs": len(spec.all_inputs),
        "val_pr_auc": val_m["pr_auc"],
        "val_roc_auc": val_m["roc_auc"],
        "val_recall_at_p90": val_m["recall_at_precision_0.90"],
        "train_pr_auc": train_m["pr_auc"],
        "gap": train_m["pr_auc"] - val_m["pr_auc"],
    }
    with mlflow.start_run(run_name=spec.name, nested=True):
        mlflow.log_params({"set": label, "n_inputs": len(spec.all_inputs), "history": spec.history})
        mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, float)})
    print(
        f"{label:52s} inputs={len(spec.all_inputs):4d}  val PR-AUC {val_m['pr_auc']:.4f}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="model config (parameters used)")
    parser.add_argument(
        "--base", type=Path, default=ROOT / "configs" / "features" / "baseline.yaml"
    )
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--only", choices=["ladder", "cuts"], default=None)
    parser.add_argument("--cuts", nargs="*", default=None, help="run only these cut labels")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "feature_sets")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()

    cfg = load_train_config(args.model)
    shipped = load_feature_spec(cfg.features)
    base = load_feature_spec(args.base)
    df = load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fraud-feature-sets")
    with mlflow.start_run(run_name="ladder_and_cuts"):
        mlflow.log_params({"git_commit": git_commit(), "model_config": args.model.name})
        if args.only in (None, "ladder"):
            rows = [evaluate(k, s, df, cfg, args.split) for k, s in ladder(base).items()]
            table = pd.DataFrame(rows)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out_dir / "ladder.csv", index=False)
            mlflow.log_text(table.to_csv(index=False), "ladder.csv")
        if args.only in (None, "cuts"):
            cuts = family_cuts(shipped, base)
            if args.cuts:
                cuts = {k: v for k, v in cuts.items() if k in args.cuts}
            rows = [evaluate(k, s, df, cfg, args.split) for k, s in cuts.items()]
            table = pd.DataFrame(rows)
            previous = args.out_dir / "family_cuts.csv"
            if args.cuts and previous.exists():  # merge partial re-runs into the full table
                old = pd.read_csv(previous)
                table = pd.concat([old[~old["set"].isin(table["set"])], table], ignore_index=True)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out_dir / "family_cuts.csv", index=False)
            mlflow.log_text(table.to_csv(index=False), "family_cuts.csv")
    print(json.dumps({"out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
