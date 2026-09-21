"""Automated retraining tests (ADR 0012): when a cycle runs, what it is judged on,
and what a promotion is allowed to change without a human.

The expensive part — that a retrained model is better — is measured offline
(`docs/retraining.md`). What these protect is the machinery around it: that an
unattended job cannot start a cycle it has no new data for, cannot compare a
challenger against a champion on the champion's own training rows, cannot promote
a tie, and cannot put a model into service under a policy nobody reviewed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import pytest
import yaml

from fraud.data import schema
from fraud.data.load import load_train
from fraud.evaluate.policy import load_policy_config
from fraud.evaluate.threshold import load_threshold_config
from fraud.serve.parity import verify
from fraud.train.cycle import month_frame, run_cycle, training_frame
from fraud.train.lifecycle import load_retrain_config
from fraud.train.promotion import load_promotion_config
from fraud.train.serving_config import read_setting, update_serving_config
from fraud.train.trigger import CyclePolicy, TriggerState, decide, load_cycle_policy

ROOT = Path(__file__).resolve().parents[1]
POLICY = CyclePolicy(
    month_days=30,
    label_maturity_days=0,
    min_new_train_days=30,
    promotion_margin=0.005,
    monitor_pr_auc_drop=0.03,
    allow_reporting_window=False,
    reporting_window_start_day=153,
)


# --- the trigger --------------------------------------------------------------------


def test_the_shipped_cycle_policy_loads() -> None:
    policy = load_cycle_policy(ROOT / "configs" / "retrain.yaml")
    assert policy.month_days > 0 and policy.min_new_train_days > 0
    assert policy.promotion_margin > 0  # an unattended job must not promote a tie
    assert policy.allow_reporting_window is False  # ADR 0002


def test_no_label_feed_means_no_cycle() -> None:
    d = decide(TriggerState(feed_clock_day=None, champion_trained_through=122), POLICY)
    assert d.run is False and d.rule == "no-labels"


def test_a_month_of_new_training_data_starts_a_cycle() -> None:
    d = decide(TriggerState(feed_clock_day=152, champion_trained_through=92), POLICY)
    assert d.run and d.rule == "calendar"
    assert (d.train_end, d.month) == (122, (123, 152))
    assert d.new_train_days == 30


def test_less_than_a_month_of_new_data_waits() -> None:
    d = decide(TriggerState(feed_clock_day=152, champion_trained_through=122), POLICY)
    assert d.run is False and d.rule == "too-little-new-data" and d.new_train_days == 0


def test_degradation_brings_a_cycle_forward_but_only_with_new_data() -> None:
    drifting = TriggerState(
        feed_clock_day=152, champion_trained_through=110, monitor_pr_auc_delta=-0.05
    )
    assert decide(drifting, POLICY).rule == "monitor"
    # …and never without: refitting the champion's own rows cannot answer drift
    stuck = replace(drifting, champion_trained_through=122)
    d = decide(stuck, POLICY)
    assert d.run is False and "cannot answer drift" in d.reason
    # a drop that has not reached the threshold is not a trigger either
    mild = replace(drifting, monitor_pr_auc_delta=-0.01)
    assert decide(mild, POLICY).run is False


def test_the_reporting_window_is_not_consumed_by_a_schedule() -> None:
    d = decide(TriggerState(feed_clock_day=183, champion_trained_through=92), POLICY)
    assert d.run is False and d.rule == "reporting-window"
    assert "ADR 0002" in d.reason
    allowed = decide(
        TriggerState(feed_clock_day=183, champion_trained_through=92),
        replace(POLICY, allow_reporting_window=True),
    )
    assert allowed.run and allowed.month == (154, 183)


def test_bootstrap_and_force() -> None:
    assert decide(TriggerState(152, None), POLICY).rule == "bootstrap"
    assert decide(TriggerState(152, 122, forced=True), POLICY).rule == "forced"


# --- the cycle ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def light_model_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The production recipe with the trees turned down: this tests wiring, not accuracy."""
    path = tmp_path_factory.mktemp("cfg") / "light.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": "fixture",
                "run_name": "fixture_model",
                "features": "configs/features/v2_freq.yaml",
                "seed": 0,
                "threshold": 0.5,
                "model": {"type": "xgboost", "params": {"n_estimators": 15, "max_depth": 3}},
            }
        )
    )
    return path


def _cycle(
    df: Any, light_model_config: Path, out_dir: Path, as_of: int, champion: Any = None,
    champion_trained_through: int | None = None, policy: CyclePolicy = POLICY,
) -> Any:  # fmt: skip
    retrain = replace(
        load_retrain_config(ROOT / "configs" / "retrain.yaml"), model_config=light_model_config
    )
    return run_cycle(
        df,
        retrain,
        policy,
        load_promotion_config(ROOT / "configs" / "promotion.yaml"),
        load_threshold_config(ROOT / "configs" / "threshold.yaml"),
        load_policy_config(ROOT / "configs" / "policy.yaml"),
        as_of,
        out_dir,
        champion,
        "fixture_champion" if champion is not None else None,
        champion_trained_through,
    )


def test_a_cycle_judges_both_models_on_a_month_neither_was_fit_on(
    fixture_raw_dir: Path, light_model_config: Path, tmp_path: Path
) -> None:
    df = load_train(fixture_raw_dir)
    stale = _cycle(df, light_model_config, tmp_path / "stale", as_of=122)
    fresh = _cycle(
        df,
        light_model_config,
        tmp_path / "fresh",
        as_of=152,
        champion=joblib.load(stale.challenger_artifact),
        champion_trained_through=stale.train_end,
    )
    assert fresh.month == (123, 152) and fresh.train_end == 122
    assert fresh.champion_trained_through == 92 and fresh.champion_metrics is not None
    # the challenger's training rows end before the month both are scored on (G3)
    train = training_frame(df, fresh.train_end)
    month = month_frame(df, fresh.month)
    assert train[schema.TIME_COL].max() // 86_400 < month[schema.TIME_COL].min() // 86_400
    assert fresh.n_month == len(month) and fresh.positives == int(month[schema.TARGET_COL].sum())
    # both models are measured with the same keys, each under its own re-derived policy
    assert set(fresh.challenger_metrics) == set(fresh.champion_metrics)
    assert fresh.challenger_metrics["block_threshold"] > 0
    # and the artifact that could be promoted reproduces its golden (G8)
    model = joblib.load(fresh.challenger_artifact)
    assert verify(model, fresh.challenger_artifact).n_rows > 0


def test_a_champion_is_never_scored_on_its_own_training_rows(
    fixture_raw_dir: Path, light_model_config: Path, tmp_path: Path
) -> None:
    """The comparison the automation must refuse rather than get wrong."""
    df = load_train(fixture_raw_dir)
    earlier = _cycle(df, light_model_config, tmp_path / "c", as_of=122)
    with pytest.raises(ValueError, match="inside the evaluation month"):
        _cycle(
            df, light_model_config, tmp_path, as_of=152,
            champion=joblib.load(earlier.challenger_artifact),
            champion_trained_through=130,
        )  # fmt: skip


def test_a_tie_is_not_promoted(
    fixture_raw_dir: Path, light_model_config: Path, tmp_path: Path
) -> None:
    """Same recipe, same rows, same seed: the challenger cannot be better, so it must
    not replace a model in service. The ADR 0010 gates alone would allow it."""
    df = load_train(fixture_raw_dir)
    first = _cycle(df, light_model_config, tmp_path / "a", as_of=152)
    again = _cycle(
        df,
        light_model_config,
        tmp_path / "b",
        as_of=152,
        champion=joblib.load(first.challenger_artifact),
        champion_trained_through=first.train_end - 1,  # an honest month, identical training rows
    )
    assert again.delta_pr_auc == pytest.approx(0.0, abs=1e-9)
    pr_auc_gate = next(g for g in again.gates if g.metric == "pr_auc")
    assert pr_auc_gate.passed is True  # ADR 0010's gate allows a tie (it allows −0.005) …
    assert again.margin_ok is False and again.promote is False  # … the cycle's margin does not
    assert "below the" in again.reason


# --- the reviewed half --------------------------------------------------------------


def _manifest(sha: str = "abc123def4567890", review: float = 0.07, block: float = 0.5) -> dict:
    return {
        "artifact_sha256": sha,
        "model_version": "fixture_model_through_day122+sigmoid",
        "bands": {"review": review, "block": block},
    }


def test_the_serving_patch_moves_bands_version_and_digest_together() -> None:
    text = (ROOT / "configs" / "serving.yaml").read_text()
    patched = update_serving_config(text, _manifest())
    loaded = yaml.safe_load(patched)
    assert loaded["bands"] == {"review": 0.07, "block": 0.5}
    assert loaded["model_version"] == "fixture_model_through_day122+sigmoid"
    assert loaded["champion_sha256"] == "abc123def456"
    # the comments that explain every value survive the edit
    assert text.count("#") == patched.count("#")
    assert "block ≥ 0.42" in patched or "docs/review_policy.md" in patched


def test_the_patch_adds_the_digest_when_the_config_has_none() -> None:
    text = "\n".join(
        line
        for line in (ROOT / "configs" / "serving.yaml").read_text().splitlines()
        if not line.startswith("champion_sha256:")
    )
    patched = update_serving_config(text + "\n", _manifest())
    assert yaml.safe_load(patched)["champion_sha256"] == "abc123def456"
    assert read_setting(patched, "champion_sha256") == "abc123def456"


def test_the_patch_refuses_a_config_it_does_not_recognise() -> None:
    with pytest.raises(ValueError, match="does not set"):
        update_serving_config("model_path: x\n", _manifest())


def test_a_fetched_champion_must_be_the_one_the_bands_were_reviewed_for(
    served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host picks the artifact, this file carries the policy: they must agree."""
    import shutil

    from fraud.serve import app as serve_app

    source = Path(served["config"]).parents[1] / "models" / "m.joblib"
    target_dir = tmp_path / "models"
    cfg = tmp_path / "serving.yaml"
    text = (
        Path(served["config"])
        .read_text()
        .replace("model_path: models/m.joblib", f"model_path: {target_dir / 'm.joblib'}")
    )
    cfg.write_text(f"{text}audit_db: null\nchampion_sha256: 0badc0de\n")

    def fake_fetch(model_path: Path, base_url: str, token: str | None, digest: str) -> list[str]:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        for src in (source, *source.parent.glob("m_frozen_*.json")):
            shutil.copyfile(src, model_path.parent / src.name)
        return ["model.joblib"]

    monkeypatch.setattr(serve_app, "fetch_champion", fake_fetch)
    monkeypatch.setattr(serve_app, "expected_champion_digest", lambda url, pin: "0badc0de")
    monkeypatch.setenv("FRAUD_CHAMPION_URL", "https://example.invalid/champion-0badc0de")
    with pytest.raises(RuntimeError, match="not the 0badc0de"):
        serve_app.load_state(cfg)


def test_force_does_not_open_the_reporting_window() -> None:
    """--force skips the calendar rules; it is not a way past ADR 0002 or missing labels."""
    forced = TriggerState(feed_clock_day=183, champion_trained_through=92, forced=True)
    assert decide(forced, POLICY).rule == "reporting-window"
    assert decide(TriggerState(None, 92, forced=True), POLICY).rule == "no-labels"


# --- what the scheduled job reads before it decides ---------------------------------


def test_the_job_reads_its_state_from_the_artifacts_that_hold_it(tmp_path: Path) -> None:
    """The three inputs to the trigger, each from the place that actually owns it."""
    import json as _json

    from scripts.retrain_cycle import feed_clock_day, monitor_delta, trained_through

    assert trained_through(None) is None
    assert trained_through({"model_info": {"training_window_days": [1, 122]}}) == 122
    assert trained_through({"model_info": {}}) is None

    assert monitor_delta(None) is None
    report = tmp_path / "m.json"
    report.write_text(_json.dumps({"model": {"eventual": {"delta_pr_auc_vs_reference": -0.042}}}))
    assert monitor_delta(report) == pytest.approx(-0.042)
    report.write_text(_json.dumps({"model": {"eventual": None}}))
    assert monitor_delta(report) is None  # a report without matured labels says nothing

    from fraud.serve.audit import AuditLog

    db = tmp_path / "audit.sqlite"
    assert feed_clock_day(db) is None  # no database at all
    audit = AuditLog(db)
    assert feed_clock_day(db) is None  # a database the feed has never run against
    audit.record_feed_run(as_of_dt=200 * 86_400 + 86_399, n_new=10, source="test")
    assert feed_clock_day(db) == 200


def test_a_failed_bootstrap_does_not_claim_a_champion_was_kept(
    fixture_raw_dir: Path, light_model_config: Path, tmp_path: Path
) -> None:
    """On the fixture the absolute gates cannot pass; the wording must not invent a
    champion to fall back on."""
    from scripts.retrain_cycle import _verdict, render

    result = _cycle(load_train(fixture_raw_dir), light_model_config, tmp_path, as_of=152)
    assert result.champion_run_name is None and result.promote is False
    assert _verdict(result, promoted=False) == "REJECT (no champion exists)"
    text = render(result, decide(TriggerState(152, None), POLICY), promoted=False)
    assert "REJECT (no champion exists)" in text and "KEEP THE CHAMPION" not in text
