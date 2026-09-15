"""Random hyperparameter search with expanding-window CV inside the training window.

Example:
    uv run python scripts/tune.py --config configs/tuning/xgboost.yaml --dev
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.train.tune import load_tune_config, run_search

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--dev", action="store_true", help="use configs/dev.yaml (10 %% sample)")
    parser.add_argument("--n-trials", type=int, default=None, help="override the config")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()

    split_cfg = load_split_config(ROOT / "configs" / "dev.yaml" if args.dev else args.split)
    tune_cfg = load_tune_config(args.config)
    if args.n_trials is not None:
        tune_cfg = type(tune_cfg)(**{**tune_cfg.__dict__, "n_trials": args.n_trials})
    df = load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed")
    train_df = split(df, split_cfg)["train"]  # validation and test never enter the search
    table = run_search(tune_cfg, split_cfg, train_df, args.tracking_uri)
    out = ROOT / "reports" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{args.config.stem}_trials.csv", index=False)
    print(table.head(5).to_string())


if __name__ == "__main__":
    main()
