"""One retraining cycle: challenger, champion, the same month, the same gates (ADR 0012).

`fraud.train.lifecycle` replays many cut-offs offline to show the lifecycle works.
This runs *one* cut-off against the artifact that is actually serving, and is what
the scheduled job calls.

The one thing it may not inherit from `scripts/promote.py` is the window. Those
gates are measured on the frozen validation window (days 123–152, ADR 0002),
which is the right window for a candidate trained on the frozen training window
and the wrong one for a model retrained *through* it: a retrained challenger has
seen those rows, so its numbers there would be training scores. So:

    train_end = mature_through − month_days      the challenger trains on days ≤ train_end
    month     = train_end+1 … mature_through     neither model has been fit on it
    both models are scored on that month, each under its own re-derived policy

The champion's stored manifest metrics are never reused for the comparison — they
were measured on a different window. It is re-scored here, on this month, or there
is no comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud.data import schema
from fraud.evaluate.policy import PolicyConfig
from fraud.evaluate.threshold import ThresholdConfig
from fraud.features.columns import load_feature_spec
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.parity import choose_sample, freeze
from fraud.train.lifecycle import SECONDS_PER_DAY, RetrainConfig, fit_challenger
from fraud.train.promotion import (
    GateResult,
    PromotionConfig,
    all_pass,
    candidate_metrics,
    run_gates,
)
from fraud.train.run import load_train_config
from fraud.train.trigger import CyclePolicy


@dataclass(frozen=True)
class CycleResult:
    """What one cycle decided, and everything it decided it from."""

    as_of_day: int
    mature_through: int
    train_end: int
    month: tuple[int, int]
    n_month: int
    positives: int
    challenger_run_name: str
    challenger_artifact: Path
    calibration: str  # the map the challenger actually carries, not an assumption
    challenger_metrics: dict[str, float]
    champion_run_name: str | None
    champion_trained_through: int | None
    champion_metrics: dict[str, float] | None
    gates: list[GateResult] = field(default_factory=list)
    gates_passed: bool = False
    margin_ok: bool = False
    promote: bool = False
    reason: str = ""

    @property
    def delta_pr_auc(self) -> float | None:
        if self.champion_metrics is None:
            return None
        return self.challenger_metrics["pr_auc"] - self.champion_metrics["pr_auc"]


def month_frame(df: pd.DataFrame, month: tuple[int, int]) -> pd.DataFrame:
    day = df[schema.TIME_COL] // SECONDS_PER_DAY
    return df.loc[(day >= month[0]) & (day <= month[1])]


def training_frame(df: pd.DataFrame, train_end: int) -> pd.DataFrame:
    return df.loc[(df[schema.TIME_COL] // SECONDS_PER_DAY) <= train_end]


def run_cycle(
    df: pd.DataFrame,
    retrain: RetrainConfig,
    policy: CyclePolicy,
    promotion: PromotionConfig,
    threshold_cfg: ThresholdConfig,
    policy_cfg: PolicyConfig,
    as_of_day: int,
    out_dir: Path,
    champion: CalibratedModel | None = None,
    champion_run_name: str | None = None,
    champion_trained_through: int | None = None,
    seed: int | None = None,
) -> CycleResult:
    """Fit a challenger for ``as_of_day``, score both models on the month, run the gates.

    Raises ``ValueError`` when the champion was trained on part of the evaluation
    month: the comparison would be the champion's training scores against the
    challenger's honest ones, which is worse than no comparison at all.
    """
    mature_through = as_of_day - policy.label_maturity_days
    train_end = mature_through - policy.month_days
    month = (train_end + 1, mature_through)
    if champion is not None and champion_trained_through is not None:
        if champion_trained_through >= month[0]:
            raise ValueError(
                f"the champion was trained through day {champion_trained_through}, inside the "
                f"evaluation month {month[0]}–{month[1]}: it would be scored on its own "
                "training rows. Advance the as-of day, or compare on a later month."
            )
    month_rows = month_frame(df, month)
    if month_rows.empty:
        raise ValueError(f"no rows in the evaluation month {month[0]}–{month[1]}")

    tc = load_train_config(retrain.model_config)
    spec = load_feature_spec(tc.features)
    seed = tc.seed if seed is None else seed
    challenger = fit_challenger(df, spec, tc.model, seed, train_end, retrain)

    run_name = f"{tc.run_name}_through_day{train_end}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"{run_name}_calibrated.joblib"
    joblib.dump(challenger, artifact)
    # The golden comes from the month, so a serving artifact is always pinned to rows
    # the model was not fit on (fraud.serve.parity, G8).
    freeze(challenger, artifact, choose_sample(month_rows, n_per_group=25))

    def measure(model: CalibratedModel) -> dict[str, float]:
        """Both models, on the same rows, each under its own re-derived policy."""
        return candidate_metrics(model, month_rows, threshold_cfg, policy_cfg, promotion)

    challenger_metrics = measure(challenger)
    champion_metrics = None if champion is None else measure(champion)
    gates = run_gates(challenger_metrics, champion_metrics, promotion.gates)
    passed = all_pass(gates)

    if champion_metrics is None:
        margin_ok = True
        reason = "bootstrap: no champion to beat"
    else:
        delta = challenger_metrics["pr_auc"] - champion_metrics["pr_auc"]
        margin_ok = delta >= policy.promotion_margin
        reason = (
            f"challenger PR-AUC {delta:+.4f} vs champion on days {month[0]}–{month[1]}"
            f" ({'clears' if margin_ok else 'below'} the {policy.promotion_margin} margin)"
        )
    if not passed:
        failed = ", ".join(g.metric for g in gates if not g.passed)
        reason = f"rejected on {failed}; {reason}"
    return CycleResult(
        as_of_day=as_of_day,
        mature_through=mature_through,
        train_end=train_end,
        month=month,
        n_month=int(len(month_rows)),
        positives=int(month_rows[schema.TARGET_COL].sum()),
        challenger_run_name=run_name,
        challenger_artifact=artifact,
        calibration=challenger.method,
        challenger_metrics=challenger_metrics,
        champion_run_name=champion_run_name,
        champion_trained_through=champion_trained_through,
        champion_metrics=champion_metrics,
        gates=gates,
        gates_passed=passed,
        margin_ok=margin_ok,
        promote=passed and margin_ok,
        reason=reason,
    )


def cycle_info(result: CycleResult, retrain: RetrainConfig, experiment: str) -> dict[str, Any]:
    """The manifest's `model_info` for a retrained champion.

    `test_pr_auc` is deliberately absent: a model retrained through the end of the
    matured data has no untouched test window left to report one on, and carrying the
    previous champion's number forward would attribute it to an artifact that never
    earned it (`/model-info` reports null).
    """
    tc = load_train_config(retrain.model_config)
    return {
        "model": tc.model["type"],
        "experiment": experiment,
        "feature_set": Path(tc.features).stem,
        "primary_metric": "pr_auc",
        "calibration": result.calibration,
        "training_window_days": [1, result.train_end],
        "validation_window_days": [result.month[0], result.month[1]],
        "retrained_as_of_day": result.as_of_day,
    }
