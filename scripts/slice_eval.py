"""Score fitted models on a named slice of the validation window (E025).

Re-scores models that are already fitted; it fits nothing and touches no test
data. The slice is given on the command line so the definition that produced a
number is recorded next to it (``docs/experiments.md`` E025).

Example:
    uv run python scripts/slice_eval.py \
        --pair xgb_f5_capacity:xgb_f7_capacity \
        --pair xgb_f5_capacity_seed1:xgb_f7_capacity_seed1 \
        --slice "ProductCD == 'W' and D1 >= 14"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.features.columns import load_feature_spec
from fraud.features.history import ENTITY_DEFINITIONS, add_entity_history
from fraud.train.run import load_train_config

ROOT = Path(__file__).resolve().parents[1]


def validation_frame(
    model_cfg: Path, split_cfg: Path, raw_dir: Path, cache_dir: Path
) -> pd.DataFrame:
    """The validation window exactly as run_experiment builds it for this config."""
    cfg = load_train_config(model_cfg)
    spec = load_feature_spec(cfg.features)
    df = load_train(raw_dir, cache_dir=cache_dir)
    if spec.history:
        df = add_entity_history(df, ENTITY_DEFINITIONS[spec.history_entity])
    return split(df, load_split_config(split_cfg))["validation"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="baseline_run_name:candidate_run_name (repeatable, one per seed)",
    )
    parser.add_argument("--slice", required=True, help="pandas query over the validation frame")
    parser.add_argument("--target", default="isFraud")
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs" / "model")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()

    config_dir: Path = args.config_dir

    def config_for(run_name: str) -> Path:
        for path in sorted(config_dir.glob("*.yaml")):
            if load_train_config(path).run_name == run_name:
                return path
        raise SystemExit(f"no config in {config_dir} has run_name {run_name!r}")

    rows: list[dict[str, object]] = []
    for pair in args.pair:
        base_name, cand_name = pair.split(":", 1)
        scored: dict[str, dict[str, float]] = {}
        for run_name in (base_name, cand_name):
            val = validation_frame(config_for(run_name), args.split, args.raw_dir, args.cache_dir)
            sliced = val.query(args.slice)
            model = joblib.load(args.models_dir / f"{run_name}.joblib")
            scored[run_name] = {
                "all": float(
                    average_precision_score(val[args.target], model.predict_proba(val)[:, 1])
                ),
                "slice": float(
                    average_precision_score(sliced[args.target], model.predict_proba(sliced)[:, 1])
                ),
                "n": float(len(sliced)),
                "pos": float(sliced[args.target].sum()),
            }
        rows.append(
            {
                "pair": pair,
                "n": scored[base_name]["n"],
                "pos": scored[base_name]["pos"],
                "base_all": scored[base_name]["all"],
                "cand_all": scored[cand_name]["all"],
                "base_slice": scored[base_name]["slice"],
                "cand_slice": scored[cand_name]["slice"],
                "delta_all": scored[cand_name]["all"] - scored[base_name]["all"],
                "delta_slice": scored[cand_name]["slice"] - scored[base_name]["slice"],
            }
        )

    out = pd.DataFrame(rows)
    print(f"\nslice: {args.slice}")
    print(f"rows in slice: {int(out['n'].iloc[0]):,}   positives: {int(out['pos'].iloc[0]):,}\n")
    cols = ["pair", "base_slice", "cand_slice", "delta_slice", "base_all", "cand_all", "delta_all"]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nmean seed-paired delta on the slice: {out['delta_slice'].mean():+.4f}")
    print(f"mean seed-paired delta overall:      {out['delta_all'].mean():+.4f}")


if __name__ == "__main__":
    main()
