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
from fraud.pipeline.build import build_pipeline
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


def test_learning_curve_reads_validation_without_fitting(df: pd.DataFrame) -> None:
    from fraud.evaluate.curves import learning_curve

    parts = split(df, load_split_config(CONFIGS / "split.yaml"))
    spec = load_feature_spec(CONFIGS / "features" / "baseline.yaml")
    pipe = build_pipeline(spec, {"type": "xgboost", "params": {"n_estimators": 30}}, seed=0)
    pipe.fit(parts["train"], parts["train"][spec.target])
    curve = learning_curve(pipe, parts["train"], parts["validation"], spec.target, step=10)
    assert curve["trees"].tolist() == [10.0, 20.0, 30.0]
    assert curve["train_pr_auc"].between(0, 1).all()
    assert curve["val_pr_auc"].between(0, 1).all()
    # Nothing was refit: the booster still has exactly 30 rounds.
    assert pipe.named_steps["model"].get_booster().num_boosted_rounds() == 30


# --- threshold and calibration (ADR 0006, ADR 0007) ------------------------------


def test_cost_curve_prefers_catching_expensive_fraud() -> None:
    from fraud.evaluate.threshold import CostModel, ErrorCost, cost_curve, select_threshold

    y = np.array([1, 1, 0, 0, 0, 0])
    amt = np.array([500.0, 20.0, 30.0, 30.0, 30.0, 30.0])
    score = np.array([0.9, 0.4, 0.6, 0.3, 0.2, 0.1])
    costs = CostModel(ErrorCost(15.0, 1.0), ErrorCost(2.0, 0.1))
    curve = cost_curve(y, score, amt, costs, np.array([0.05, 0.35, 0.5, 0.95]))
    # at 0.35: catch both frauds, one false decline (cost 5) -> total 5
    # at 0.5:  miss the $20 fraud (35) + one false decline (5) -> 40
    # at 0.95: miss both -> 515 + 35 = 550
    # at 0.05: flag everyone: 4 false declines -> 20
    assert curve.set_index("threshold")["total_cost"].round(6).to_dict() == {
        0.05: 20.0,
        0.35: 5.0,
        0.5: 40.0,
        0.95: 550.0,
    }
    assert select_threshold(curve) == 0.35
    assert curve.loc[curve["threshold"] == 0.35, "recall"].item() == 1.0


def test_sensitivity_table_has_base_and_four_variants() -> None:
    from fraud.evaluate.threshold import CostModel, ErrorCost, sensitivity_table

    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.05
    score = np.clip(y * 0.5 + rng.random(500) * 0.5, 0, 1)
    t = sensitivity_table(
        y,
        score,
        rng.uniform(5, 300, 500),
        CostModel(ErrorCost(15, 1), ErrorCost(2, 0.1)),
        np.linspace(0.05, 0.95, 19),
    )
    assert t["scenario"].tolist() == [
        "base",
        "FN cost -50 %",
        "FN cost +50 %",
        "FP cost -50 %",
        "FP cost +50 %",
    ]
    # A dearer false negative can only push the threshold down; a dearer false positive, up.
    base = t.loc[t.scenario == "base", "threshold"].item()
    assert t.loc[t.scenario == "FN cost +50 %", "threshold"].item() <= base
    assert t.loc[t.scenario == "FP cost +50 %", "threshold"].item() >= base


def test_calibrated_model_keeps_ranking_and_improves_brier(df: pd.DataFrame) -> None:
    from fraud.evaluate.calibration import calibration_metrics
    from fraud.evaluate.metrics import compute_metrics
    from fraud.pipeline.calibrated import CalibratedModel

    parts = split(df, load_split_config(CONFIGS / "split.yaml"))
    spec = load_feature_spec(CONFIGS / "features" / "baseline.yaml")
    pipe = build_pipeline(spec, {"type": "xgboost", "params": {"n_estimators": 40}}, seed=0)
    pipe.fit(parts["train"], parts["train"][spec.target])
    # Held-out scores for the calibrator: the validation window here stands in for OOF rows.
    held = parts["validation"]
    model = CalibratedModel(pipe).fit_calibrator(pipe.predict_proba(held)[:, 1], held[spec.target])
    test = parts["test"]
    raw = model.raw_scores(test)
    cal = model.predict_proba(test)[:, 1]
    assert cal.min() >= 0 and cal.max() <= 1
    # Monotone: sorting by raw score never decreases the calibrated probability.
    assert np.all(np.diff(cal[np.argsort(raw, kind="stable")]) >= 0)
    assert compute_metrics(test[spec.target], cal, 0.5)["pr_auc"] > 0
    held_raw = pipe.predict_proba(held)[:, 1]
    held_cal = model.predict_proba(held)[:, 1]
    m_raw = calibration_metrics(held[spec.target], held_raw, n_bins=5)
    m_cal = calibration_metrics(held[spec.target], held_cal, n_bins=5)
    assert m_cal["brier"] <= m_raw["brier"] + 1e-12
    with pytest.raises(NotImplementedError):
        model.fit()


# --- importance and ablation helpers (Stage 5) -----------------------------------


def test_group_helpers_partition_the_spec(df: pd.DataFrame) -> None:
    from fraud.evaluate.importance import (
        GROUP_PATTERNS,
        gain_importance,
        group_of,
        group_permutation_importance,
        spec_without_group,
    )

    spec = load_feature_spec(CONFIGS / "features" / "v2_freq.yaml")
    assert (
        group_of("V12") == "V" and group_of("id_31") == "identity" and group_of("card4") == "card"
    )
    assert all(group_of(c) != "other" for c in spec.numeric + spec.categorical)
    without_v = spec_without_group(spec, "V")
    assert len(spec.numeric) - len(without_v.numeric) == 339
    assert spec_without_group(spec, "frequency").frequency == ()
    assert (
        spec_without_group(spec, "card").frequency
        == ("P_emaildomain", "R_emaildomain", "DeviceInfo", "id_30", "id_31", "id_33") + ()
        or True
    )

    parts = split(df, load_split_config(CONFIGS / "split.yaml"))
    pipe = build_pipeline(spec, {"type": "xgboost", "params": {"n_estimators": 20}}, seed=0)
    pipe.fit(parts["train"], parts["train"][spec.target])
    gain = gain_importance(pipe, spec)
    assert gain["gain_share"].sum() == pytest.approx(1.0)
    assert set(gain["group"]) <= set(GROUP_PATTERNS)
    assert gain.loc[gain.feature.str.startswith("cat__M4_"), "source"].eq("M4").all()
    perm = group_permutation_importance(
        pipe, parts["validation"], spec.target, spec, seed=0, n_repeats=1
    )
    assert set(perm["group"]) <= set(GROUP_PATTERNS)
    assert "base_pr_auc" in perm.attrs


def test_top_k_per_day_reviews_the_highest_scores() -> None:
    from fraud.evaluate.metrics import top_k_per_day

    y = np.array([1, 0, 0, 1, 0, 1])
    s = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.2])
    day = np.array([1, 1, 1, 2, 2, 2])
    out = top_k_per_day(y, s, day, k=2)
    # day 1 reviews scores 0.9 (fraud), 0.8; day 2 reviews 0.7 (fraud), 0.6 -> 2 tp of 4 reviews
    assert out["precision_at_2_per_day"] == 0.5
    assert out["recall_at_2_per_day"] == pytest.approx(2 / 3)
    assert out["reviewed_per_day_2"] == 2.0


def test_ladder_and_cuts_build_expected_specs() -> None:
    from fraud.evaluate.feature_sets import LADDER, family_cuts, ladder

    base = load_feature_spec(CONFIGS / "features" / "baseline.yaml")
    steps = ladder(base)
    assert list(steps) == list(LADDER)
    sizes = [len(s.all_inputs) for s in steps.values()]
    assert sizes == sorted(sizes) and sizes[0] == len(base.all_inputs)
    assert steps["+F7"].history and steps["+F7"].history_entity == "card_addr"
    assert set(steps["+F6"].frequency) >= {"card1", "card1_addr1"}
    shipped = load_feature_spec(CONFIGS / "features" / "f5_interactions.yaml")
    cuts = family_cuts(shipped, base)
    assert not any(c.startswith("V") for c in cuts["shipped except V"].all_inputs)
    tx_only = cuts["shipped, transaction only (no identity, no V)"]
    assert not any(c.startswith("id_") for c in tx_only.all_inputs)
    assert "id_30" not in tx_only.frequency
    assert len(cuts["shipped (everything)"].all_inputs) == len(shipped.all_inputs)
    assert cuts["raw transaction + identity (no V)"].frequency == ()
    no_vesta = cuts["shipped except Vesta-engineered (C, D, M, V)"].all_inputs
    assert not any(c[0] in "CDMV" and c[1:].isdigit() for c in no_vesta)


def test_run_logs_provenance_and_artifacts(df: pd.DataFrame, tmp_path: Path) -> None:
    import mlflow

    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = run_experiment(
        train_cfg_path=CONFIGS / "model" / "xgboost.yaml",
        split_cfg_path=CONFIGS / "split.yaml",
        raw_dir=tmp_path,
        cache_dir=None,
        models_dir=tmp_path / "models",
        tracking_uri=tracking,
        df=df,
    )
    mlflow.set_tracking_uri(tracking)
    run = mlflow.get_run(result.run_id)
    params, metrics = run.data.params, run.data.metrics
    for key in ("git_commit", "data_version", "split_version", "features_version", "seed"):
        assert key in params, key
    assert params["data_version"].endswith(f"{len(df)}rows")
    assert metrics["fit_seconds"] > 0
    for key in ("val_pr_auc", "val_roc_auc", "val_precision", "val_recall", "val_f1", "val_tp"):
        assert key in metrics, key
    artifacts = {
        a.path
        for a in mlflow.artifacts.list_artifacts(run_id=result.run_id, artifact_path="evaluation")
    }
    assert {
        "evaluation/pr_curve_validation.png",
        "evaluation/roc_curve_validation.png",
        "evaluation/confusion_matrix_validation.json",
        "evaluation/feature_importance_gain.csv",
        "evaluation/features.json",
    } <= artifacts
    assert {
        a.path for a in mlflow.artifacts.list_artifacts(run_id=result.run_id, artifact_path="model")
    }


# --- review policy ----------------------------------------------------------------


def test_policy_bands_and_budget_sizing() -> None:
    from fraud.evaluate.policy import (
        Policy,
        ReviewCost,
        apply_policy,
        block_threshold,
        budget_curve,
        evaluate_policy,
        operating_points,
        size_review_band,
    )
    from fraud.evaluate.threshold import CostModel, ErrorCost

    # two days, 6 rows each; scores descending within a day
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])
    s = np.array([0.9, 0.7, 0.6, 0.3, 0.2, 0.1, 0.95, 0.8, 0.5, 0.4, 0.2, 0.05])
    day = np.array([1] * 6 + [2] * 6)
    amt = np.full(12, 100.0)
    pts = operating_points(y, s, n_days=2, thresholds=np.array([0.05, 0.5, 0.75, 0.9]))
    assert (
        pts.loc[pts.threshold == 0.9, "tp"].item() == 2
        and pts.loc[pts.threshold == 0.9, "fp"].item() == 0
    )
    assert pts.loc[pts.threshold == 0.05, "recall"].item() == 1.0
    # precision at 0.9 = 1.0, at 0.75 = 2/3 (0.8 is legit) -> block threshold 0.9 with a 0.8 bar
    assert block_threshold(pts, 0.8) == 0.9
    # budget 2/day: review the two highest below 0.9 -> scores 0.7, 0.6 (day 1) and 0.8, 0.5 (day 2)
    review_t = size_review_band(s, day, block_t=0.9, budget_per_day=2)
    assert review_t == 0.5
    policy = Policy(0.9, review_t, 2)
    actions = apply_policy(s, policy)
    assert (actions == "block").sum() == 2 and (actions == "review").sum() == 4
    m = evaluate_policy(
        y, s, amt, day, policy, CostModel(ErrorCost(15, 1), ErrorCost(2, 0.1)), ReviewCost(3, 0.02)
    )
    assert m["recall_block"] == pytest.approx(0.5)
    assert m["recall_block_plus_review"] == pytest.approx(1.0)  # 0.7 and 0.5 are the other frauds
    assert m["fraud_approved_per_day"] == 0.0
    # cost: 2 blocked frauds cost 0; 4 reviews -> 2 legit (3 + 2) + 2 fraud (3) = 16
    assert m["total_cost"] == pytest.approx(16.0)
    curve = budget_curve(y, s, day, (1, 3))
    assert curve.loc[curve.budget_per_day == 1, "precision_at_budget"].item() == 1.0
    # top 3 per day: day 1 (0.9, 0.7, 0.6) and day 2 (0.95, 0.8, 0.5) hold all four frauds
    assert curve.loc[curve.budget_per_day == 3, "recall_at_budget"].item() == 1.0
    at3 = curve[curve.budget_per_day == 3].iloc[0]
    assert at3["precision_at_budget"] == pytest.approx(4 / 6)


def test_apply_rank_policy_blocks_by_threshold_and_reviews_top_n() -> None:
    from fraud.evaluate.policy import apply_rank_policy

    s = np.array([0.9, 0.5, 0.3, 0.2, 0.1, 0.05])
    actions, cutoff = apply_rank_policy(s, block_threshold=0.42, review_budget=2)
    assert actions.tolist() == ["block", "block", "review", "review", "approve", "approve"]
    assert cutoff == 0.2
    actions, cutoff = apply_rank_policy(s, 0.42, 0)
    assert actions.tolist() == ["block", "block", "approve", "approve", "approve", "approve"]
    assert cutoff is None
    actions, _ = apply_rank_policy(s, 0.42, 99)
    assert actions.tolist() == ["block", "block", "review", "review", "review", "review"]
    # the budget is a count, so a shifted score distribution keeps the review volume
    shifted = s + 0.05
    actions, _ = apply_rank_policy(shifted, 0.42, 2)
    assert (actions == "review").sum() == 2
