"""Model and leakage tests on the fixture: beats the prior, nothing fit on validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud.data import schema
from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.features.columns import load_feature_spec
from fraud.pipeline.baseline import build_pipeline
from fraud.train.run import TrainConfig, fit_and_evaluate, run_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="module")
def df(fixture_raw_dir: Path) -> pd.DataFrame:
    return load_train(fixture_raw_dir)


def _cfg(model: dict[str, object], seed: int = 42) -> TrainConfig:
    return TrainConfig(
        experiment="test",
        run_name="test",
        features=CONFIGS / "features" / "baseline.yaml",
        seed=seed,
        threshold=0.5,
        model=model,
    )


def test_logreg_beats_constant_on_validation(df: pd.DataFrame) -> None:
    parts = split(df, load_split_config(CONFIGS / "split.yaml"))
    spec = load_feature_spec(CONFIGS / "features" / "baseline.yaml")
    _, _, const = fit_and_evaluate(parts, spec, _cfg({"type": "constant"}))
    _, _, lr = fit_and_evaluate(parts, spec, _cfg({"type": "logreg", "params": {"max_iter": 500}}))
    assert const["pr_auc"] == pytest.approx(parts["validation"][schema.TARGET_COL].mean(), abs=1e-6)
    assert lr["pr_auc"] > const["pr_auc"]
    assert lr["roc_auc"] > 0.5


def test_probabilities_in_unit_interval(df: pd.DataFrame) -> None:
    parts = split(df, load_split_config(CONFIGS / "split.yaml"))
    spec = load_feature_spec(CONFIGS / "features" / "baseline.yaml")
    pipe = build_pipeline(spec, {"type": "logreg", "params": {"max_iter": 500}}, seed=1)
    pipe.fit(parts["train"], parts["train"][spec.target])
    p = pipe.predict_proba(parts["test"])[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_nothing_is_fit_on_validation_or_test(df: pd.DataFrame) -> None:
    """Corrupting every non-training row must not change the fitted pipeline (G2)."""
    cfg = load_split_config(CONFIGS / "split.yaml")
    spec = load_feature_spec(CONFIGS / "features" / "baseline.yaml")
    clean = split(df, cfg)

    corrupted = df.copy()
    later = corrupted[schema.TIME_COL] >= cfg.start_seconds("validation")
    corrupted.loc[later, schema.TARGET_COL] = 1 - corrupted.loc[later, schema.TARGET_COL]
    corrupted.loc[later, "TransactionAmt"] *= 1000
    corrupted.loc[later, "ProductCD"] = "CORRUPT"
    corrupted.loc[later, list(schema.V_COLS)] = np.nan
    dirty = split(corrupted, cfg)
    pd.testing.assert_frame_equal(clean["train"], dirty["train"])

    model = {"type": "logreg", "params": {"max_iter": 500}}
    pipe_clean, _, _ = fit_and_evaluate(clean, spec, _cfg(model))
    pipe_dirty, _, _ = fit_and_evaluate(dirty, spec, _cfg(model))

    imp_clean = pipe_clean.named_steps["features"].named_transformers_["num"]["impute"]
    imp_dirty = pipe_dirty.named_steps["features"].named_transformers_["num"]["impute"]
    np.testing.assert_array_equal(imp_clean.statistics_, imp_dirty.statistics_)
    ohe_clean = pipe_clean.named_steps["features"].named_transformers_["cat"]["onehot"]
    ohe_dirty = pipe_dirty.named_steps["features"].named_transformers_["cat"]["onehot"]
    for a, b in zip(ohe_clean.categories_, ohe_dirty.categories_, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(
        pipe_clean.named_steps["model"].coef_, pipe_dirty.named_steps["model"].coef_
    )
    # And scores on the untouched test rows are identical.
    np.testing.assert_allclose(
        pipe_clean.predict_proba(clean["test"])[:, 1], pipe_dirty.predict_proba(clean["test"])[:, 1]
    )


def test_run_is_reproducible(df: pd.DataFrame, tmp_path: Path) -> None:
    """Two runs from the same config and seed give the same validation metric."""
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    kwargs = dict(
        train_cfg_path=CONFIGS / "model" / "logreg.yaml",
        split_cfg_path=CONFIGS / "split.yaml",
        raw_dir=tmp_path,
        cache_dir=None,
        models_dir=tmp_path / "models",
        tracking_uri=tracking,
        df=df,
    )
    a = run_experiment(**kwargs)  # type: ignore[arg-type]
    b = run_experiment(**kwargs)  # type: ignore[arg-type]
    assert a.validation_metrics["pr_auc"] == pytest.approx(b.validation_metrics["pr_auc"], abs=1e-9)
    assert a.model_path.exists()
    assert a.run_id != b.run_id
