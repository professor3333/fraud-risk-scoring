"""Rolling temporal backtests with horizons for one or more candidates (ADR 0008).

Example:
    uv run python scripts/backtest.py --candidates configs/model/xgboost_f5_capacity.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from fraud.data.load import load_train
from fraud.evaluate.backtest import horizons, load_backtest_config, run_candidate, summarise
from fraud.features.columns import load_feature_spec
from fraud.train.run import git_commit, load_train_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "backtest.yaml")
    parser.add_argument("--seed", type=int, default=None, help="override every candidate's seed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "backtest")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()

    cfg = load_backtest_config(args.config)
    df = load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"T{h.train_end}+{h.gap}d [{h.start}-{h.end}]" for h in horizons(cfg)]
    print("horizons:", ", ".join(labels))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    tables = []
    with mlflow.start_run(run_name="rolling_backtest"):
        mlflow.log_params(
            {"git_commit": git_commit(), "config": args.config.name, "seed_override": args.seed}
        )
        for path in args.candidates:
            tc = load_train_config(path)
            seed = tc.seed if args.seed is None else args.seed
            spec = load_feature_spec(tc.features)
            with mlflow.start_run(run_name=f"{tc.run_name}_seed{seed}", nested=True):
                params = {f"model_{k}": v for k, v in tc.model.get("params", {}).items()}
                mlflow.log_params({"candidate": tc.run_name, "seed": seed, **params})
                table = run_candidate(tc.run_name, df, spec, tc.model, seed, cfg)
                for _, r in table.iterrows():
                    key = f"pr_auc_T{int(r.train_end)}_gap{int(r.gap)}"
                    mlflow.log_metric(key, float(r.pr_auc))
                s = summarise(table, cfg.robust_min_gap).iloc[0]
                mlflow.log_metrics(
                    {
                        "adjacent_pr_auc": s.adjacent_pr_auc,
                        "robust_pr_auc": s.robust_pr_auc,
                        "decay": s.decay,
                    }
                )
            table["seed"] = seed
            tables.append(table)
            print(
                f"{tc.run_name} (seed {seed}): adjacent {s.adjacent_pr_auc:.4f}  "
                f"robust(gap>={cfg.robust_min_gap}) {s.robust_pr_auc:.4f}  decay {s.decay:.4f}",
                flush=True,
            )
        results = pd.concat(tables, ignore_index=True)
        suffix = "" if args.seed is None else f"_seed{args.seed}"
        results.to_csv(args.out_dir / f"windows{suffix}.csv", index=False)
        summary = summarise(results, cfg.robust_min_gap)
        summary.to_csv(args.out_dir / f"summary{suffix}.csv", index=False)
        mlflow.log_text(results.to_csv(index=False), "windows.csv")
        mlflow.log_text(summary.to_csv(index=False), "summary.csv")
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
