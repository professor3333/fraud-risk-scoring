"""E021: deliberate one-parameter-at-a-time XGBoost sweeps (underfit -> capacity -> overfit -> regularise).

Anchored on a model config; each axis varies one parameter with the rest fixed.
Reports train and validation PR-AUC and the gap so the arc is visible. The
tree-count axis is read post hoc from one long fit. Every fit is an MLflow run.

Example:
    uv run python scripts/param_sweep.py --config configs/tuning/sweep.yaml
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.curves import learning_curve
from fraud.features.columns import load_feature_spec
from fraud.pipeline.build import build_pipeline
from fraud.train.run import TrainConfig, fit_and_evaluate, git_commit, load_train_config

ROOT = Path(__file__).resolve().parents[1]


def one_fit(
    label: str, params: dict[str, Any], parts: dict[str, pd.DataFrame], cfg: TrainConfig, spec: Any
) -> dict[str, Any]:
    run_cfg = replace(cfg, model={**cfg.model, "params": {**cfg.model["params"], **params}})
    t0 = time.perf_counter()
    _, train_m, val_m = fit_and_evaluate(parts, spec, run_cfg)
    seconds = time.perf_counter() - t0
    row = {
        "axis": label.split("=")[0],
        "setting": label,
        **params,
        "train_pr_auc": train_m["pr_auc"],
        "val_pr_auc": val_m["pr_auc"],
        "gap": train_m["pr_auc"] - val_m["pr_auc"],
        "val_roc_auc": val_m["roc_auc"],
        "val_recall_at_p90": val_m["recall_at_precision_0.90"],
        "fit_seconds": seconds,
    }
    with mlflow.start_run(run_name=label, nested=True):
        mlflow.log_params({f"model_{k}": v for k, v in run_cfg.model["params"].items()})
        mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, float)})
    print(
        f"{label:26s} train {train_m['pr_auc']:.4f}  val {val_m['pr_auc']:.4f}  "
        f"gap {row['gap']:.3f}  ({seconds:.0f}s)",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "sweeps")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--skip-trees", action="store_true")
    args = parser.parse_args()

    sweep = yaml.safe_load(args.config.read_text())
    cfg = load_train_config(ROOT / sweep["model_config"])
    spec = load_feature_spec(cfg.features)
    anchor = dict(cfg.model["params"])
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(sweep["experiment"])
    with mlflow.start_run(run_name="one_at_a_time"):
        mlflow.log_params({"git_commit": git_commit(), "anchor": str(anchor)})
        rows: list[dict[str, Any]] = []
        rows.append(one_fit("anchor", {}, parts, cfg, spec))
        for axis, values in sweep["axes"].items():
            for v in values:
                if v == anchor.get(axis):
                    continue  # already measured as the anchor
                rows.append(one_fit(f"{axis}={v}", {axis: v}, parts, cfg, spec))
        for name in ("overfit", "regularised"):
            rows.append(one_fit(name, dict(sweep[name]["params"]), parts, cfg, spec))
        table = pd.DataFrame(rows)
        table.to_csv(args.out_dir / "one_at_a_time.csv", index=False)
        mlflow.log_text(table.to_csv(index=False), "one_at_a_time.csv")

        if not args.skip_trees:
            n, step = int(sweep["trees"]["n_estimators"]), int(sweep["trees"]["step"])
            long_cfg = replace(cfg, model={**cfg.model, "params": {**anchor, "n_estimators": n}})
            pipe = build_pipeline(spec, long_cfg.model, long_cfg.seed)
            pipe.fit(parts["train"], parts["train"][spec.target])
            curve = learning_curve(pipe, parts["train"], parts["validation"], spec.target, step)
            curve.to_csv(args.out_dir / "trees.csv", index=False)
            mlflow.log_text(curve.to_csv(index=False), "trees.csv")
            print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
