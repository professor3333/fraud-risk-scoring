"""When a retraining cycle should run, and why (ADR 0012).

The cycle itself is expensive and its result is a model that may be deployed, so
the decision to start one is separated from the work and kept pure: state in, a
:class:`TriggerDecision` out, no files and no clock of its own.

Three things can start a cycle:

- **the calendar** — the label feed has matured a full month of labels the
  champion was not trained on;
- **the monitor** — eventual performance has fallen below the reference by more
  than the configured drop (`docs/monitoring.md`), *and* there is new data to
  train on. Retraining on exactly the champion's rows cannot fix drift, so
  degradation alone is never enough;
- **an operator**, with `--force`.

And one thing stops one: the evaluation month would reach into the reporting
window (ADR 0002). That is a consultation of the held-out window, so it happens
only when `allow_reporting_window` is set in a reviewed commit, never because a
scheduled job's clock rolled forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CyclePolicy:
    """configs/retrain.yaml → `cycle:`. What starts a cycle and what a cycle must beat."""

    month_days: int
    label_maturity_days: int
    min_new_train_days: int
    promotion_margin: float
    monitor_pr_auc_drop: float
    allow_reporting_window: bool
    reporting_window_start_day: int

    def __post_init__(self) -> None:
        if self.month_days <= 0 or self.min_new_train_days <= 0:
            raise ValueError("month_days and min_new_train_days must be positive")
        if self.promotion_margin < 0 or self.monitor_pr_auc_drop < 0:
            raise ValueError("promotion_margin and monitor_pr_auc_drop must not be negative")


def load_cycle_policy(path: Path) -> CyclePolicy:
    raw = yaml.safe_load(path.read_text())
    cycle = raw["cycle"]
    return CyclePolicy(
        month_days=int(raw["month_days"]),
        label_maturity_days=int(cycle.get("label_maturity_days", raw["label_maturity_days"])),
        min_new_train_days=int(cycle["min_new_train_days"]),
        promotion_margin=float(cycle["promotion_margin"]),
        monitor_pr_auc_drop=float(cycle["monitor_pr_auc_drop"]),
        allow_reporting_window=bool(cycle["allow_reporting_window"]),
        reporting_window_start_day=int(cycle["reporting_window_start_day"]),
    )


@dataclass(frozen=True)
class TriggerState:
    """Everything the decision reads, gathered by the caller."""

    feed_clock_day: int | None  # the day the label feed has advanced to; None = never fed
    champion_trained_through: int | None  # last training day of the served model; None = none yet
    monitor_pr_auc_delta: float | None = None  # eventual PR-AUC minus its reference
    forced: bool = False


@dataclass(frozen=True)
class TriggerDecision:
    run: bool
    rule: str  # calendar | monitor | bootstrap | forced | no-labels | too-little-new-data |
    #            reporting-window
    reason: str
    as_of_day: int | None = None
    mature_through: int | None = None
    train_end: int | None = None
    month: tuple[int, int] | None = None
    new_train_days: int | None = None


def decide(state: TriggerState, policy: CyclePolicy) -> TriggerDecision:
    """Whether to run a cycle now, at which as-of day, and on what grounds."""
    if state.feed_clock_day is None:
        return TriggerDecision(
            False,
            "no-labels",
            "the label feed has never run, so no window of matured labels exists "
            "(scripts/feedback.py or POST /outcomes)",
        )
    as_of = int(state.feed_clock_day)
    mature_through = as_of - policy.label_maturity_days
    train_end = mature_through - policy.month_days
    month = (train_end + 1, mature_through)
    new_days = (
        None
        if state.champion_trained_through is None
        else train_end - int(state.champion_trained_through)
    )

    def verdict(run: bool, rule: str, reason: str) -> TriggerDecision:
        return TriggerDecision(run, rule, reason, as_of, mature_through, train_end, month, new_days)

    if train_end < policy.month_days:
        return verdict(
            False,
            "no-labels",
            f"labels are mature only through day {mature_through}; that leaves nothing to "
            "train on before the evaluation month",
        )
    if mature_through >= policy.reporting_window_start_day and not policy.allow_reporting_window:
        return verdict(
            False,
            "reporting-window",
            f"the evaluation month {month[0]}–{month[1]} reaches into the reporting window "
            f"(day {policy.reporting_window_start_day}+). Consuming it is a consultation of "
            "the held-out window (ADR 0002): set cycle.allow_reporting_window in a reviewed "
            "commit and record it in the log",
        )
    if new_days is None:
        return verdict(True, "bootstrap", "there is no champion to compare against")
    if state.forced:
        return verdict(True, "forced", f"requested by an operator ({new_days} new training days)")
    if new_days >= policy.min_new_train_days:
        return verdict(
            True,
            "calendar",
            f"{new_days} training days have matured since the champion's cut-off "
            f"(day {state.champion_trained_through}), at or past the "
            f"{policy.min_new_train_days}-day step",
        )
    drop = state.monitor_pr_auc_delta
    degraded = drop is not None and drop <= -policy.monitor_pr_auc_drop
    if degraded and new_days > 0:
        return verdict(
            True,
            "monitor",
            f"eventual PR-AUC is {drop:+.3f} against the reference (past "
            f"−{policy.monitor_pr_auc_drop:.3f}) and {new_days} new training days exist",
        )
    if degraded:
        return verdict(
            False,
            "too-little-new-data",
            f"eventual PR-AUC is {drop:+.3f} against the reference, but no training day has "
            "matured since the champion's cut-off: refitting the same rows cannot answer drift",
        )
    return verdict(
        False,
        "too-little-new-data",
        f"only {new_days} new training days since the champion's cut-off "
        f"(day {state.champion_trained_through}); the step is {policy.min_new_train_days}",
    )
