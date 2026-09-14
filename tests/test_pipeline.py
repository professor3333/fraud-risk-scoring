"""Pipeline tests: fit on a fixture, transform unseen rows, stable shapes, save/load parity."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud.data import schema
from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.features.columns import FeatureSpec, load_feature_spec
from fraud.pipeline.baseline import MISSING_LEVEL, build_pipeline

ROOT = Path(__file__).resolve().parents[1]
LOGREG = {"type": "logreg", "params": {"C": 1.0, "max_iter": 500}}


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return load_feature_spec(ROOT / "configs" / "features" / "baseline.yaml")


@pytest.fixture(scope="module")
def parts(fixture_raw_dir: Path) -> dict[str, pd.DataFrame]:
    df = load_train(fixture_raw_dir)
    return split(df, load_split_config(ROOT / "configs" / "split.yaml"))


def test_feature_spec_excludes_forbidden_columns(spec: FeatureSpec) -> None:
    assert schema.TARGET_COL not in spec.all_inputs
    assert schema.ID_COL not in spec.all_inputs
    assert schema.TIME_COL not in spec.all_inputs
    assert len(spec.numeric) == 9 + 14 + 15 + 339 + 23 + 1
    assert "has_identity" in spec.numeric


def test_fit_transform_shapes_are_stable(spec: FeatureSpec, parts: dict[str, pd.DataFrame]) -> None:
    pipe = build_pipeline(spec, LOGREG, seed=0)
    train = parts["train"]
    pipe.fit(train, train[spec.target])
    x_train = pipe[:-1].transform(train)
    x_val = pipe[:-1].transform(parts["validation"])
    assert x_train.shape[1] == x_val.shape[1]
    assert x_train.dtype == np.float64
    assert not np.isnan(x_val).any()


def test_unseen_categories_and_missing_identity(
    spec: FeatureSpec, parts: dict[str, pd.DataFrame]
) -> None:
    pipe = build_pipeline(spec, LOGREG, seed=0)
    train = parts["train"]
    pipe.fit(train, train[spec.target])
    rows = parts["validation"].head(5).copy()
    rows["ProductCD"] = "ZZ"  # never seen in training
    rows["card4"] = None
    idn_cols = [c for c in schema.IDENTITY_COLS if c != schema.ID_COL]
    rows[idn_cols] = None  # identity absent
    rows["has_identity"] = False
    scores = pipe.predict_proba(rows)[:, 1]
    assert scores.shape == (5,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_missing_level_is_a_category(spec: FeatureSpec, parts: dict[str, pd.DataFrame]) -> None:
    pipe = build_pipeline(spec, LOGREG, seed=0)
    train = parts["train"]
    pipe.fit(train, train[spec.target])
    names = list(pipe.named_steps["features"].get_feature_names_out())
    assert any(n.endswith(f"_{MISSING_LEVEL}") for n in names)
    assert any(n.startswith("num__missingindicator_") for n in names)


def test_save_load_identical_predictions(
    spec: FeatureSpec, parts: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    pipe = build_pipeline(spec, LOGREG, seed=0)
    train = parts["train"]
    pipe.fit(train, train[spec.target])
    before = pipe.predict_proba(parts["test"])[:, 1]
    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path)
    after = joblib.load(path).predict_proba(parts["test"])[:, 1]
    np.testing.assert_array_equal(before, after)


def test_pipeline_selects_its_own_columns(
    spec: FeatureSpec, parts: dict[str, pd.DataFrame]
) -> None:
    """Extra columns in the input are ignored; a missing input column is an error."""
    pipe = build_pipeline(spec, LOGREG, seed=0)
    train = parts["train"]
    pipe.fit(train, train[spec.target])
    with_extra = parts["validation"].assign(junk=1.0)
    np.testing.assert_array_equal(
        pipe.predict_proba(with_extra)[:, 1], pipe.predict_proba(parts["validation"])[:, 1]
    )
    with pytest.raises(ValueError):
        pipe.predict_proba(parts["validation"].drop(columns=["ProductCD"]))
