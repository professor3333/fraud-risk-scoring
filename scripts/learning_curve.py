"""Plot train vs validation PR-AUC as trees are added, for a saved XGBoost pipeline.

Example:
    uv run python scripts/learning_curve.py --model models/xgb_v2_freq.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.curves import learning_curve, plot_learning_curve

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "curves")
    args = parser.parse_args()

    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    pipe = joblib.load(args.model)
    curve = learning_curve(pipe, parts["train"], parts["validation"], "isFraud", args.step)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.model.stem
    curve.to_csv(args.out_dir / f"{stem}.csv", index=False)
    plot_learning_curve(curve, stem).savefig(args.out_dir / f"{stem}.png", dpi=120)
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
