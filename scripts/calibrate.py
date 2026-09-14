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
    model = CalibratedModel(pipeline).fit_calibrator(oof_df["score"], oof_df["y"])

    raw = model.raw_scores(val)
    cal = model.predict_proba(val)[:, 1]
    y = val[spec.target].to_numpy()
    raw_m, cal_m = calibration_metrics(y, raw), calibration_metrics(y, cal)
    rank_raw, rank_cal = (
        compute_metrics(y, raw, 0.5)["pr_auc"],
        compute_metrics(y, cal, 0.5)["pr_auc"],
    )
    # Isotonic steps tie some scores, so ranking metrics may move slightly; large moves are a bug.
    if abs(rank_raw - rank_cal) > 0.01:
        raise RuntimeError(f"calibration changed the ranking too much: {rank_raw} vs {rank_cal}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(args.out_dir / f"{cfg.run_name}_oof.csv", index=False)
    tables = {"raw": reliability_table(y, raw), "isotonic (OOF-fit)": reliability_table(y, cal)}
    fig = plot_reliability(tables, f"{cfg.run_name} / validation")
    fig.savefig(args.out_dir / f"{cfg.run_name}_reliability.png", dpi=120)
    summary = {
        "raw": raw_m,
        "calibrated": cal_m,
        "val_pr_auc_raw": rank_raw,
        "val_pr_auc_calibrated": rank_cal,
        "n_oof": int(len(oof_df)),
    }
    (args.out_dir / f"{cfg.run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    out_model = args.models_dir / f"{cfg.run_name}_calibrated.joblib"
    joblib.dump(model, out_model)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fraud-calibration")
    with mlflow.start_run(run_name=f"{cfg.run_name}_isotonic"):
        mlflow.log_params(
            {"git_commit": git_commit(), "base_run_name": cfg.run_name, "n_oof": len(oof_df)}
        )
        mlflow.log_metrics({f"raw_{k}": v for k, v in raw_m.items()})
        mlflow.log_metrics({f"cal_{k}": v for k, v in cal_m.items()})
        mlflow.log_artifacts(str(args.out_dir), "calibration")
        mlflow.log_artifact(str(out_model), "model")

    print(json.dumps(summary, indent=2))
    print(f"saved {out_model}")


if __name__ == "__main__":
    main()
