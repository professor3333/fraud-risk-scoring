"""Hyperparameter search with time-ordered expanding-window CV (ADR 0002, §8).

Folds live entirely inside the training window. Each trial fits one pipeline
per fold on the fold's training days and scores the fold's later days; the
trial's score is the mean fold average precision. Every trial is logged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml

from fraud.data import schema
from fraud.data.split import SplitConfig
from fraud.evaluate.metrics import compute_metrics
from fraud.features.columns import FeatureSpec, load_feature_spec
from fraud.pipeline.build import build_pipeline
from fraud.train.run import git_commit


@dataclass(frozen=True)
class Fold:
    train_end_day: int
    val_start_day: int
    val_end_day: int

    def __post_init__(self) -> None:
        if not self.train_end_day < self.val_start_day <= self.val_end_day:
            raise ValueError(f"fold days must be ordered: {self}")


@dataclass(frozen=True)
class TuneConfig:
    experiment: str
    features: Path
    seed: int
    n_trials: int
    folds: tuple[Fold, ...]
    fixed: dict[str, Any]
    space: dict[str, list[Any]]
    reference: dict[str, Any] | None = None


def load_tune_config(path: Path) -> TuneConfig:
    raw = yaml.safe_load(path.read_text())
    return TuneConfig(
        experiment=str(raw["experiment"]),
        features=Path(raw["features"]),
        seed=int(raw["seed"]),
        n_trials=int(raw["n_trials"]),
        folds=tuple(Fold(**f) for f in raw["folds"]),
        fixed=dict(raw.get("fixed", {})),
        space={str(k): list(v) for k, v in raw["space"].items()},
        reference=dict(raw["reference"]) if raw.get("reference") else None,
    )


def check_folds_inside_training_window(folds: tuple[Fold, ...], split_cfg: SplitConfig) -> None:
    train = split_cfg.window("train")
    for f in folds:
        if f.val_end_day > train.end_day or f.train_end_day < train.start_day:
            raise ValueError(f"fold {f} leaves the training window {train}")


def fold_frames(
    train_df: pd.DataFrame, fold: Fold, seconds_per_day: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = train_df[schema.TIME_COL] // seconds_per_day
    fit = train_df.loc[day <= fold.train_end_day]
    score = train_df.loc[(day >= fold.val_start_day) & (day <= fold.val_end_day)]
    if fit[schema.TIME_COL].max() >= score[schema.TIME_COL].min():
        raise ValueError(f"fold {fold}: fit rows are not strictly earlier than score rows")
    return fit, score


def sample_params(space: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    return {k: rng.choice(v) for k, v in space.items()}


def cv_score(
    train_df: pd.DataFrame,
    spec: FeatureSpec,
    params: dict[str, Any],
    folds: tuple[Fold, ...],
    seconds_per_day: int,
    seed: int,
) -> list[float]:
    scores: list[float] = []
    for fold in folds:
        fit, score = fold_frames(train_df, fold, seconds_per_day)
        pipe = build_pipeline(spec, {"type": "xgboost", "params": params}, seed)
        pipe.fit(fit, fit[spec.target])
        m = compute_metrics(score[spec.target], pipe.predict_proba(score)[:, 1], 0.5)
        scores.append(m["pr_auc"])
    return scores


def run_search(
    tune_cfg: TuneConfig,
    split_cfg: SplitConfig,
    train_df: pd.DataFrame,
    tracking_uri: str,
) -> pd.DataFrame:
    """Random search; returns one row per trial sorted by mean fold PR-AUC, best first."""
    check_folds_inside_training_window(tune_cfg.folds, split_cfg)
    spec = load_feature_spec(tune_cfg.features)
    rng = random.Random(tune_cfg.seed)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(tune_cfg.experiment)
    rows: list[dict[str, Any]] = []
    with mlflow.start_run(run_name="random_search") as parent:
        mlflow.log_params(
            {
                "git_commit": git_commit(),
                "seed": tune_cfg.seed,
                "n_trials": tune_cfg.n_trials,
                "feature_spec": spec.name,
                "folds": ";".join(
                    f"{f.train_end_day}/{f.val_start_day}-{f.val_end_day}" for f in tune_cfg.folds
                ),
            }
        )
        trials: list[tuple[str, dict[str, Any]]] = []
        if tune_cfg.reference is not None:
            trials.append(("reference", {**tune_cfg.fixed, **tune_cfg.reference}))
        trials += [
            (f"trial_{i:02d}", {**tune_cfg.fixed, **sample_params(tune_cfg.space, rng)})
            for i in range(tune_cfg.n_trials)
        ]
        for name, params in trials:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({f"model_{k}": v for k, v in params.items()})
                scores = cv_score(
                    train_df, spec, params, tune_cfg.folds, split_cfg.seconds_per_day, tune_cfg.seed
                )
                mean, std = float(np.mean(scores)), float(np.std(scores))
                mlflow.log_metrics(
                    {"cv_pr_auc_mean": mean, "cv_pr_auc_std": std}
                    | {f"fold{j}_pr_auc": s for j, s in enumerate(scores)}
                )
            rows.append({"trial": name, "cv_pr_auc_mean": mean, "cv_pr_auc_std": std, **params})
            print(f"{name}  cv_pr_auc={mean:.4f} ±{std:.4f}  {params}", flush=True)
        table = pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False)
        mlflow.log_metric("best_cv_pr_auc", float(table["cv_pr_auc_mean"].iloc[0]))
        mlflow.log_text(table.to_csv(index=False), "trials.csv")
        print(f"parent run: {parent.info.run_id}")
    return table.reset_index(drop=True)
