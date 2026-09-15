"""Fit an isotonic calibrator on out-of-fold training-window scores and wrap the saved pipeline.

Out-of-fold scores come from the expanding-window folds in the tuning config
(fit on days <= train_end_day, score the fold's later days), so the calibrator
never sees validation or test. Validation is then used to *assess* raw vs
calibrated probabilities (Brier, ECE, reliability diagram).

Example:
    uv run python scripts/calibrate.py --model-config configs/model/xgboost_v2_tuned.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.calibration import calibration_metrics, plot_reliability, reliability_table
from fraud.evaluate.metrics import compute_metrics
from fraud.features.columns import load_feature_spec
from fraud.pipeline.build import build_pipeline
from fraud.pipeline.calibrated import CalibratedModel
from fraud.train.run import git_commit, load_train_config
from fraud.train.tune import fold_frames, load_tune_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--tuning-config", type=Path, default=ROOT / "configs" / "tuning" / "xgboost.yaml"
    )
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "calibration")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args()

    cfg = load_train_config(args.model_config)
    spec = load_feature_spec(cfg.features)
    split_cfg = load_split_config(args.split)
    folds = load_tune_config(args.tuning_config).folds
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"), split_cfg
    )
    train, val = parts["train"], parts["validation"]

    # Out-of-fold scores inside the training window.
    oof: list[pd.DataFrame] = []
    for fold in folds:
        fit, score = fold_frames(train, fold, split_cfg.seconds_per_day)
        pipe = build_pipeline(spec, cfg.model, cfg.seed)
        pipe.fit(fit, fit[spec.target])
        oof.append(
            pd.DataFrame(
                {
                    spec.id_col: score[spec.id_col].to_numpy(),
                    "TransactionAmt": score["TransactionAmt"].to_numpy(),
                    "y": score[spec.target].to_numpy(),
                    "score": pipe.predict_proba(score)[:, 1],
                }
            )
        )
        print(f"fold {fold}: {len(score)} out-of-fold rows", flush=True)
    oof_df = pd.concat(oof, ignore_index=True)

    pipeline = joblib.load(args.models_dir / f"{cfg.run_name}.joblib")
    y = val[spec.target].to_numpy()
    raw = pipeline.predict_proba(val)[:, 1]
    raw_m = calibration_metrics(y, raw)
    rank_raw = compute_metrics(y, raw, 0.5)["pr_auc"]

    candidates: dict[str, CalibratedModel] = {}
    results: dict[str, dict[str, float]] = {"raw": {**raw_m, "val_pr_auc": rank_raw}}
    tables = {"raw": reliability_table(y, raw)}
    for method in ("sigmoid", "isotonic"):
        m = CalibratedModel(pipeline, method=method).fit_calibrator(oof_df["score"], oof_df["y"])
        cal = m.calibrate_scores(raw)
        results[method] = {
            **calibration_metrics(y, cal),
            "val_pr_auc": compute_metrics(y, cal, 0.5)["pr_auc"],
        }
        tables[method] = reliability_table(y, cal)
        candidates[method] = m

    # ADR 0007 rule: better Brier and ECE than raw, and PR-AUC within 0.01 of raw.
    passing = [
        m
        for m in ("sigmoid", "isotonic")
        if results[m]["brier"] < raw_m["brier"]
        and results[m]["ece"] < raw_m["ece"]
        and abs(results[m]["val_pr_auc"] - rank_raw) <= 0.01
    ]
    chosen = min(passing, key=lambda m: results[m]["brier"]) if passing else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(args.out_dir / f"{cfg.run_name}_oof.csv", index=False)
    fig = plot_reliability(tables, f"{cfg.run_name} / validation")
    fig.savefig(args.out_dir / f"{cfg.run_name}_reliability.png", dpi=120)
    summary = {"results": results, "chosen": chosen, "n_oof": int(len(oof_df))}
    (args.out_dir / f"{cfg.run_name}_summary.json").write_text(json.dumps(summary, indent=2))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fraud-calibration")
    with mlflow.start_run(run_name=f"{cfg.run_name}_calibration"):
        mlflow.log_params(
            {
                "git_commit": git_commit(),
                "base_run_name": cfg.run_name,
                "n_oof": len(oof_df),
                "chosen": str(chosen),
            }
        )
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})
        mlflow.log_artifacts(str(args.out_dir), "calibration")
        if chosen is not None:
            out_model = args.models_dir / f"{cfg.run_name}_calibrated.joblib"
            joblib.dump(candidates[chosen], out_model)
            mlflow.log_artifact(str(out_model), "model")
            print(f"saved {out_model} ({chosen})")
        else:
            print("no calibration method passed the ADR 0007 rule; raw scores stay")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
