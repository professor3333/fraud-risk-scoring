"""Train a configured model on the training window and log the run to MLflow.

Examples:
    uv run python scripts/train.py --model configs/model/logreg.yaml --dev
    uv run python scripts/train.py --model configs/model/logreg.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud.train.run import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="model config YAML")
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--dev", action="store_true", help="use configs/dev.yaml (10 %% sample)")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()

    split_cfg = ROOT / "configs" / "dev.yaml" if args.dev else args.split
    result = run_experiment(
        train_cfg_path=args.model,
        split_cfg_path=split_cfg,
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        models_dir=args.models_dir,
        tracking_uri=args.tracking_uri,
    )
    print(f"run_id: {result.run_id}")
    print(f"model:  {result.model_path}")
    keys = ("pr_auc", "roc_auc", "precision", "recall", "f1", "recall_at_precision_0.90")
    print(f"{'metric':28s} {'train':>10s} {'validation':>12s}")
    for k in keys:
        print(f"{k:28s} {result.train_metrics[k]:10.4f} {result.validation_metrics[k]:12.4f}")
    cm = result.validation_metrics
    cells = " ".join(f"{k}={cm[k]:.0f}" for k in ("tp", "fp", "fn", "tn"))
    print(f"validation confusion: {cells}")


if __name__ == "__main__":
    main()
