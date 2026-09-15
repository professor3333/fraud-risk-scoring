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
from fraud.pipeline.build import MISSING_LEVEL, build_pipeline

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


def test_tree_preprocessing_keeps_nan_and_xgboost_scores(
    spec: FeatureSpec, parts: dict[str, pd.DataFrame]
) -> None:
    xgb = {"type": "xgboost", "params": {"n_estimators": 20, "max_depth": 3}}
    pipe = build_pipeline(spec, xgb, seed=0)
    train = parts["train"]
    pipe.fit(train, train[spec.target])
    x_val = pipe[:-1].transform(parts["validation"])
    assert np.isnan(x_val).any()  # trees route NaN natively; nothing is imputed
    assert x_val.shape[1] == len(spec.numeric) + sum(
        len(c)
        for c in pipe.named_steps["features"].named_transformers_["cat"]["onehot"].categories_
    )
    p = pipe.predict_proba(parts["test"])[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_v2_freq_pipeline_fits_tables_on_train_only(parts: dict[str, pd.DataFrame]) -> None:
    spec2 = load_feature_spec(ROOT / "configs" / "features" / "v2_freq.yaml")
    assert spec2.frequency == (
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "id_30",
        "id_31",
        "id_33",
    )
    xgb = {"type": "xgboost", "params": {"n_estimators": 10, "max_depth": 2}}
    pipe = build_pipeline(spec2, xgb, seed=0)
    train = parts["train"]
    pipe.fit(train, train[spec2.target])
    enc = pipe.named_steps["features"].named_transformers_["freq"]
    assert enc.n_fit_ == len(train)
    assert sum(enc.tables_["card1"].values()) == pytest.approx(1.0)
    names = list(pipe.named_steps["features"].get_feature_names_out())
    assert "freq__freq_card1" in names
    p = pipe.predict_proba(parts["validation"])[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_rare_levels_and_unseen_levels_share_one_column(parts: dict[str, pd.DataFrame]) -> None:
    from dataclasses import replace

    spec_rare = replace(
        load_feature_spec(ROOT / "configs" / "features" / "baseline.yaml"), rare_min_frequency=55
    )
    pipe = build_pipeline(spec_rare, {"type": "xgboost", "params": {"n_estimators": 5}}, seed=0)
    train = parts["train"]
    pipe.fit(train, train[spec_rare.target])
    names = list(pipe.named_steps["features"].get_feature_names_out())
    # ProductCD's five fixture levels have 51-62 training rows; those under 55 get grouped.
    assert "cat__ProductCD_infrequent_sklearn" in names
    assert "cat__ProductCD_C" in names and "cat__ProductCD_S" not in names
    rows = parts["validation"].head(3).copy()
    rows["ProductCD"] = "NEVER_SEEN"
    x = pipe[:-1].transform(rows)
    idx = [
        i
        for i, n in enumerate(pipe.named_steps["features"].get_feature_names_out())
        if n == "cat__ProductCD_infrequent_sklearn"
    ]
    assert len(idx) == 1
    assert np.all(x[:, idx[0]] == 1.0)


def test_preprocessing_override_puts_imputation_in_front_of_trees(
    parts: dict[str, pd.DataFrame], spec: FeatureSpec
) -> None:
    cfg = {"type": "xgboost", "preprocessing": "linear", "params": {"n_estimators": 5}}
    pipe = build_pipeline(spec, cfg, seed=0)
    train = parts["train"]
    pipe.fit(train, train[spec.target])
    num = pipe.named_steps["features"].named_transformers_["num"]
    assert "impute" in num.named_steps
    assert not np.isnan(pipe[:-1].transform(parts["validation"])).any()
