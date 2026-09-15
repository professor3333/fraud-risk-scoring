"""Prediction parity (G8): training pipeline == saved artifact == API == frozen golden.

Fixture-level tests run in CI. The slow test checks the real production artifact
against the frozen sample written by scripts/freeze_artifact.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from fraud.data import schema
from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.features.columns import load_feature_spec
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.app import create_app, load_state
from fraud.serve.parity import choose_sample, freeze, frozen_paths, load_frozen_sample, verify
from fraud.train.run import fit_and_evaluate, load_train_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
GOLDEN = ROOT / "tests" / "fixtures" / "frozen_features.json"


def _payload(row: pd.Series) -> dict[str, object]:
    out: dict[str, object] = {}
    for col, val in row.items():
        if col in (schema.TARGET_COL, schema.HAS_IDENTITY_COL) or pd.isna(val):
            continue
        if col in (schema.ID_COL, schema.TIME_COL):
            out[col] = int(val)
        elif col in schema.TRANSACTION_STR_COLS | schema.IDENTITY_STR_COLS:
            out[col] = str(val)
        else:
            out[col] = float(val)
    return out


@pytest.fixture(scope="module")
def frozen_fixture(fixture_raw_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Train through the training code path, calibrate, save, freeze — all on the fixture."""
    root = tmp_path_factory.mktemp("parity")
    (root / "configs").mkdir()
    (root / "models").mkdir()
    cfg = load_train_config(CONFIGS / "model" / "xgboost.yaml")
    spec = load_feature_spec(cfg.features)
    parts = split(load_train(fixture_raw_dir), load_split_config(CONFIGS / "split.yaml"))
    pipe, _, _ = fit_and_evaluate(parts, spec, cfg)  # the training path
    training_scores = pipe.predict_proba(parts["test"])[:, 1]
    held = parts["validation"]
    model = CalibratedModel(pipe, method="sigmoid").fit_calibrator(
        pipe.predict_proba(held)[:, 1], held[spec.target]
    )
    artifact = root / "models" / "m.joblib"
    joblib.dump(model, artifact)
    sample = choose_sample(parts["test"], n_per_group=5)
    freeze(model, artifact, sample)
    serving = root / "configs" / "serving.yaml"
    serving.write_text(
        "model_path: models/m.joblib\nmodel_version: fixture\nrequire_parity: true\n"
        "bands: {review: 0.062, block: 0.42}\n"
    )
    return {
        "artifact": artifact,
        "serving": serving,
        "sample": sample,
        "test_rows": parts["test"],
        "training_scores": training_scores,
        "model": model,
    }


def test_training_path_equals_loaded_artifact_raw_scores(frozen_fixture: dict) -> None:
    loaded = joblib.load(frozen_fixture["artifact"])
    np.testing.assert_array_equal(
        loaded.raw_scores(frozen_fixture["test_rows"]), frozen_fixture["training_scores"]
    )


def test_loaded_artifact_matches_frozen_golden(frozen_fixture: dict) -> None:
    result = verify(joblib.load(frozen_fixture["artifact"]), frozen_fixture["artifact"])
    assert result.n_rows == 10 and result.max_abs_diff == 0.0


def test_api_matches_frozen_golden_and_offline(frozen_fixture: dict) -> None:
    _, expected_path = frozen_paths(frozen_fixture["artifact"])
    expected = json.loads(expected_path.read_text())["probabilities"]
    offline = frozen_fixture["model"].predict_proba(frozen_fixture["sample"])[:, 1]
    with TestClient(create_app(frozen_fixture["serving"])) as client:
        health = client.get("/health").json()
        assert health["parity_rows"] == 10
        for (_, row), off in zip(frozen_fixture["sample"].iterrows(), offline, strict=True):
            body = client.post("/predict", json=_payload(row)).json()
            tid = str(int(row[schema.ID_COL]))
            assert body["fraud_probability"] == pytest.approx(expected[tid], abs=1e-9)
            assert body["fraud_probability"] == pytest.approx(off, abs=1e-9)


def test_service_refuses_to_start_on_parity_mismatch(frozen_fixture: dict, tmp_path: Path) -> None:
    """A tampered golden, a swapped artifact, or a missing sample must stop the service."""
    _, expected_path = frozen_paths(frozen_fixture["artifact"])
    original = expected_path.read_text()
    tampered = json.loads(original)
    first = next(iter(tampered["probabilities"]))
    tampered["probabilities"][first] += 0.01
    expected_path.write_text(json.dumps(tampered))
    try:
        with pytest.raises(RuntimeError, match="parity failure"):
            load_state(frozen_fixture["serving"])
    finally:
        expected_path.write_text(original)
    # a different artifact under the same name (retrained calibrator): sha mismatch
    import copy

    other = copy.deepcopy(frozen_fixture["model"])
    other.calibrator_.intercept_ = other.calibrator_.intercept_ + 0.1
    swapped = tmp_path / "m.joblib"
    joblib.dump(other, swapped)
    frozen_paths(swapped)[0].write_bytes(frozen_paths(frozen_fixture["artifact"])[0].read_bytes())
    frozen_paths(swapped)[1].write_text(original)
    with pytest.raises(RuntimeError, match="not the one the frozen sample was made for"):
        verify(joblib.load(swapped), swapped)
    with pytest.raises(FileNotFoundError):
        verify(other, tmp_path / "nothing.joblib")


def test_fixture_feature_matrix_matches_committed_golden(frozen_fixture: dict) -> None:
    """Refitting the fixture preprocessing reproduces the committed feature matrix.

    Guards against silent changes in feature construction or preprocessing across
    machines. Model outputs are deliberately not part of this golden: XGBoost trees
    fitted on a tiny sample differ across platforms, so per-artifact goldens
    (scripts/freeze_artifact.py) pin those instead. Regenerate with
    `uv run python tests/fixtures/make_fixtures.py --golden` when a change is intended.
    """
    want = json.loads(GOLDEN.read_text())
    pipe = frozen_fixture["model"].pipeline
    names = list(pipe.named_steps["features"].get_feature_names_out())
    assert names == want["feature_names"]
    sample = frozen_fixture["sample"]
    matrix = pipe[:-1].transform(sample)
    for tid, row in zip(sample[schema.ID_COL], matrix, strict=True):
        expected = want["rows"][str(int(tid))]
        for name, got, exp in zip(names, row, expected, strict=True):
            if exp is None:
                assert np.isnan(got), (tid, name)
            else:
                assert got == pytest.approx(exp, abs=1e-8), (tid, name)


# --- the production artifact -----------------------------------------------------------


@pytest.mark.slow
def test_production_artifact_parity(full_raw_dir: Path) -> None:
    serving = yaml.safe_load((CONFIGS / "serving.yaml").read_text())
    artifact = ROOT / serving["model_path"]
    if not artifact.exists():
        pytest.skip("production artifact not built")
    model = joblib.load(artifact)
    result = verify(model, artifact)
    assert result.n_rows == 50 and result.max_abs_diff == 0.0
    # training-path pipeline (saved by scripts/train.py) == the production object's inner pipeline
    training_pipe = joblib.load(str(artifact).replace("_calibrated.joblib", ".joblib"))
    sample = load_frozen_sample(artifact)
    np.testing.assert_array_equal(
        training_pipe.predict_proba(sample)[:, 1], model.raw_scores(sample)
    )
    # API == golden
    expected = json.loads(frozen_paths(artifact)[1].read_text())["probabilities"]
    with TestClient(create_app(CONFIGS / "serving.yaml")) as client:
        assert client.get("/health").json()["parity_rows"] == 50
        for _, row in sample.iterrows():
            body = client.post("/predict", json=_payload(row)).json()
            assert body["fraud_probability"] == pytest.approx(
                expected[str(int(row[schema.ID_COL]))], abs=1e-9
            )
