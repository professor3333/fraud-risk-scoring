"""Train one configured model on the training window and log it to MLflow (G6, §8).

Validation metrics are computed here because they drive decisions. The test
window is never touched by this module; ``scripts/evaluate.py`` handles it.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from fraud.data.load import load_train
from fraud.data.split import SplitConfig, check_split, load_split_config, split
from fraud.evaluate.metrics import compute_metrics, plot_pr_curve
from fraud.features.columns import FeatureSpec, load_feature_spec
from fraud.features.history import ENTITY_DEFINITIONS, add_entity_history
from fraud.pipeline.build import build_pipeline


@dataclass(frozen=True)
class TrainConfig:
    experiment: str
    run_name: str
    features: Path
    seed: int
    threshold: float
    model: dict[str, Any] = field(default_factory=dict)


def load_train_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text())
    return TrainConfig(
        experiment=str(raw["experiment"]),
        run_name=str(raw["run_name"]),
        features=Path(raw["features"]),
        seed=int(raw["seed"]),
        threshold=float(raw["threshold"]),
        model=dict(raw["model"]),
    )


@dataclass(frozen=True)
class RunResult:
    run_id: str
    model_path: Path
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip()


def fit_and_evaluate(
    parts: dict[str, pd.DataFrame],
    spec: FeatureSpec,
    cfg: TrainConfig,
) -> tuple[Pipeline, dict[str, float], dict[str, float]]:
    """Fit on ``parts['train']`` only; score train and validation."""
    np.random.seed(cfg.seed)
    train, val = parts["train"], parts["validation"]
    pipe = build_pipeline(spec, cfg.model, cfg.seed)
    pipe.fit(train, train[spec.target])
    train_scores = pipe.predict_proba(train)[:, 1]
    val_scores = pipe.predict_proba(val)[:, 1]
    return (
        pipe,
        compute_metrics(train[spec.target], train_scores, cfg.threshold),
        compute_metrics(val[spec.target], val_scores, cfg.threshold),
    )


def run_experiment(
    train_cfg_path: Path,
    split_cfg_path: Path,
    raw_dir: Path,
    cache_dir: Path | None,
    models_dir: Path,
    tracking_uri: str,
    df: pd.DataFrame | None = None,
) -> RunResult:
    cfg = load_train_config(train_cfg_path)
    split_cfg: SplitConfig = load_split_config(split_cfg_path)
    spec = load_feature_spec(cfg.features)

    if df is None:
        df = load_train(raw_dir, cache_dir=cache_dir)
    if spec.history:
        # Strictly-earlier-rows features over the full time-ordered frame (ADR 0004).
        df = add_entity_history(df, ENTITY_DEFINITIONS[spec.history_entity])
    parts = split(df, split_cfg)
    check_split(parts, split_cfg, spec.id_col)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params(
            {
                "git_commit": git_commit(),
                "seed": cfg.seed,
                "threshold": cfg.threshold,
                "model_type": cfg.model["type"],
                "feature_spec": spec.name,
                "n_numeric": len(spec.numeric),
                "n_categorical": len(spec.categorical),
                "split_config": split_cfg_path.name,
                "sample_fraction": split_cfg.sample_fraction,
                **{f"model_{k}": v for k, v in cfg.model.get("params", {}).items()},
                **{f"split_{w.name}_days": f"{w.start_day}-{w.end_day}" for w in split_cfg.windows},
                **{f"n_{name}": len(part) for name, part in parts.items()},
            }
        )
        pipe, train_m, val_m = fit_and_evaluate(parts, spec, cfg)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
        mlflow.log_metric("gap_pr_auc", train_m["pr_auc"] - val_m["pr_auc"])

        mlflow.log_artifact(str(train_cfg_path), "config")
        mlflow.log_artifact(str(split_cfg_path), "config")
        mlflow.log_artifact(str(cfg.features), "config")
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            (tmpdir / "features.json").write_text(
                json.dumps(
                    {"numeric": list(spec.numeric), "categorical": list(spec.categorical)},
                    indent=2,
                )
            )
            (tmpdir / "confusion_matrix_validation.json").write_text(
                json.dumps({k: val_m[k] for k in ("tp", "fp", "fn", "tn")}, indent=2)
            )
            val = parts["validation"]
            fig = plot_pr_curve(
                val[spec.target], pipe.predict_proba(val)[:, 1], f"{cfg.run_name} / validation"
            )
            fig.savefig(tmpdir / "pr_curve_validation.png", dpi=120)
            mlflow.log_artifacts(str(tmpdir), "evaluation")

        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{cfg.run_name}.joblib"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path), "model")
        return RunResult(
            run_id=run.info.run_id,
            model_path=model_path,
            train_metrics=train_m,
            validation_metrics=val_m,
        )
