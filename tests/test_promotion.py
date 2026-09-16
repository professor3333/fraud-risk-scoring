"""Promotion tests: gate arithmetic, candidate metrics, the champion directory the service
reads, and the registry ledger (ADR 0010)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pytest
from fastapi.testclient import TestClient

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.policy import load_policy_config
from fraud.evaluate.threshold import load_threshold_config
from fraud.features.columns import load_feature_spec
from fraud.pipeline.build import build_pipeline
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.app import create_app
from fraud.serve.parity import choose_sample, freeze, read_manifest
from fraud.train.promotion import (
    CHAMPION_ARTIFACT,
    Gate,
    PromotionConfig,
    all_pass,
    build_manifest,
    candidate_metrics,
    champion_version,
    load_promotion_config,
    materialise,
    register_candidate,
    run_gates,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_gates_apply_bounds_and_champion_tolerance() -> None:
    gates = (
        Gate("pr_auc", True, tolerance_vs_champion=0.005),
        Gate("ece", False, max=0.01),
        Gate("recall", True, tolerance_vs_champion=0.01, min=0.6),
    )
    cand = {"pr_auc": 0.60, "ece": 0.004, "recall": 0.70}
    # no champion: only absolute bounds apply
    boot = run_gates(cand, None, gates)
    assert all_pass(boot) and all(r.champion is None for r in boot)
    # within tolerance of the champion passes; beyond it fails with the reason spelled out
    ok = run_gates(cand, {"pr_auc": 0.604, "ece": 0.001, "recall": 0.705}, gates)
    assert all_pass(ok)
    bad = run_gates(cand, {"pr_auc": 0.61, "ece": 0.001, "recall": 0.70}, gates)
    assert not all_pass(bad)
    failed = {r.metric: r.reason for r in bad if not r.passed}
    assert set(failed) == {"pr_auc"} and "champion 0.6100" in failed["pr_auc"]
    # absolute bounds fail regardless of the champion
    assert not run_gates({**cand, "ece": 0.02}, None, gates)[1].passed
    assert not run_gates({**cand, "recall": 0.5}, {"recall": 0.4}, gates)[2].passed
    # a champion missing a metric (older manifest) is compared on what it has
    assert all_pass(run_gates(cand, {"pr_auc": 0.60}, gates))


def test_promotion_config_loads() -> None:
    cfg = load_promotion_config(CONFIGS / "promotion.yaml")
    assert cfg.champion_dir == Path("models/champion")
    names = {g.metric for g in cfg.gates}
    assert {"pr_auc", "ece", "brier", "cost_per_transaction"} <= names
    assert f"recall_at_{cfg.review_budget_per_day}_per_day" in names


@pytest.fixture(scope="module")
def candidate(fixture_raw_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("promotion")
    (root / "configs").mkdir()
    (root / "models").mkdir()
    parts = split(load_train(fixture_raw_dir), load_split_config(CONFIGS / "split.yaml"))
    spec = load_feature_spec(CONFIGS / "features" / "v2_freq.yaml")
    pipe = build_pipeline(spec, {"type": "xgboost", "params": {"n_estimators": 30}}, seed=0)
    pipe.fit(parts["train"], parts["train"][spec.target])
    held = parts["validation"]
    model = CalibratedModel(pipe, method="sigmoid").fit_calibrator(
        pipe.predict_proba(held)[:, 1], held[spec.target]
    )
    artifact = root / "models" / "cand_calibrated.joblib"
    joblib.dump(model, artifact)
    freeze(model, artifact, choose_sample(parts["validation"], n_per_group=5))
    return {"root": root, "artifact": artifact, "model": model, "parts": parts}


def test_candidate_metrics_and_materialised_champion_is_what_the_service_serves(
    candidate: dict[str, Any],
) -> None:
    cfg = PromotionConfig(
        registry_model="test-scorer",
        champion_dir=candidate["root"] / "models" / "champion",
        review_budget_per_day=5,
        block_min_precision=0.5,
        gates=(Gate("pr_auc", True, min=0.0), Gate("ece", False, max=1.0)),
    )
    parts = candidate["parts"]
    metrics = candidate_metrics(
        candidate["model"], parts["validation"], load_threshold_config(CONFIGS / "threshold.yaml"),
        load_policy_config(CONFIGS / "policy.yaml"), cfg,
    )  # fmt: skip
    assert 0 <= metrics["pr_auc"] <= 1 and 0 <= metrics["recall_at_5_per_day"] <= 1
    assert 0 < metrics["review_threshold"] <= metrics["block_threshold"] < 1
    assert metrics["cost_per_transaction"] >= 0 and metrics["positives"] > 0
    gates = run_gates(metrics, None, cfg.gates)
    assert all_pass(gates)

    info = {"model": "xgboost", "experiment": "fixture", "feature_set": "v2_freq",
            "calibration": "sigmoid", "training_window_days": [1, 122],
            "validation_window_days": [123, 152]}  # fmt: skip
    manifest = build_manifest("cand", candidate["artifact"], metrics, gates, info, 1)
    target = materialise(
        candidate["artifact"], candidate["model"], cfg, manifest,
        parts["train"], parts["validation"],
    )  # fmt: skip
    assert target.name == CHAMPION_ARTIFACT
    stored = read_manifest(target)
    assert stored is not None and stored["run_name"] == "cand"
    assert stored["bands"]["block"] == metrics["block_threshold"]
    assert (cfg.champion_dir / "model_monitor_reference.json").exists()

    # the service reads its version, facts and bands from the manifest — the config names none
    serving = candidate["root"] / "configs" / "serving.yaml"
    serving.write_text(
        f"model_path: {target}\naudit_db: null\nmodel_version: stale-config-value\n"
        "bands: {review: 0.01, block: 0.99}\nmodel_info: {experiment: stale}\n"
    )
    with TestClient(create_app(serving)) as client:
        info_body = client.get("/model-info").json()
        assert info_body["version"].startswith("cand+sigmoid@")
        assert info_body["experiment"] == "fixture"
        assert info_body["bands"] == {
            "review": pytest.approx(metrics["review_threshold"]),
            "block": pytest.approx(metrics["block_threshold"]),
        }
        assert info_body["validation_pr_auc"] == pytest.approx(metrics["pr_auc"])
    # a manifest describing a different artifact is refused at startup
    target.write_bytes(candidate["artifact"].read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="not the artifact its manifest describes"):
        with TestClient(create_app(serving)):
            pass


def test_registry_records_every_candidate_and_moves_the_alias(
    candidate: dict[str, Any], tmp_path: Path
) -> None:
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    cfg = PromotionConfig("test-scorer", tmp_path / "champion", 5, 0.5, ())
    gates = [
        run_gates({"pr_auc": 0.6}, None, (Gate("pr_auc", True, min=0.0),))[0],
    ]
    assert champion_version(uri, "test-scorer") is None
    v1 = register_candidate(uri, cfg, candidate["artifact"], "first", gates, promoted=True)
    assert champion_version(uri, "test-scorer") == v1
    rejected = run_gates({"pr_auc": 0.2}, None, (Gate("pr_auc", True, min=0.5),))
    v2 = register_candidate(uri, cfg, candidate["artifact"], "second", rejected, promoted=False)
    assert v2 == v1 + 1 and champion_version(uri, "test-scorer") == v1  # alias unchanged
    from mlflow import MlflowClient

    tags = MlflowClient().get_model_version("test-scorer", v2).tags
    assert tags["verdict"] == "rejected" and "pr_auc" in tags["gates"]
