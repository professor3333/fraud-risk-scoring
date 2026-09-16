"""Delayed labels: how outcomes arrive, and what can be measured before they all have.

A fraud label is not known when a transaction is scored. A chargeback is reported
some days later; a transaction that is never reported is called legitimate only
once the reporting window (the host's 120 days, ADR 0001) has closed. So at any
moment a scored transaction is one of:

- matured:  its outcome has arrived (a report, or the window closed without one)
- pending:  no outcome yet and the window is still open — expected
- overdue:  no outcome and the window has closed — the feed has a gap

Two consequences drive everything here (ADR 0009):

1. Positives arrive before negatives. Among the labels that have arrived for a
   young cohort, fraud is over-represented, so a positive rate or a PR-AUC on
   "labels so far" is biased. Eventual performance is computed on cohorts whose
   window has closed; younger cohorts give only an early signal (how much of the
   fraud reported so far was blocked or reviewed).
2. Retraining data is the closed cohort, not the labelled rows.

The simulation supplies the report dates the dataset lacks: a positive's report
lag is log-normal (configs/feedback.yaml), a negative is confirmed at maturity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fraud.data import schema

SECONDS_PER_DAY = 86_400
STATUSES = ("matured", "pending", "overdue", "unknown_time")


@dataclass(frozen=True)
class FeedbackConfig:
    maturity_days: int
    report_lag_median_days: float
    report_lag_sigma: float
    seed: int

    @property
    def maturity_seconds(self) -> int:
        return self.maturity_days * SECONDS_PER_DAY


def load_feedback_config(path: Path) -> FeedbackConfig:
    raw = yaml.safe_load(path.read_text())
    lag = raw["report_lag"]
    return FeedbackConfig(
        maturity_days=int(raw["maturity_days"]),
        report_lag_median_days=float(lag["median_days"]),
        report_lag_sigma=float(lag["sigma"]),
        seed=int(raw["seed"]),
    )


def simulate_arrivals(labels: pd.DataFrame, cfg: FeedbackConfig) -> pd.DataFrame:
    """When each label becomes known, on the TransactionDT clock.

    `labels` has TransactionID, TransactionDT, isFraud. Positives get a log-normal
    report lag clipped to the maturity window (a fraud labelled by the host was, by
    the host's rule, reported inside it); negatives are confirmed when the window
    closes. Rows are processed in TransactionID order with one seeded generator, so
    the same transaction gets the same lag on every run whatever else is passed.
    """
    ordered = labels.sort_values(schema.ID_COL).reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed)
    lag_days = rng.lognormal(
        mean=np.log(cfg.report_lag_median_days), sigma=cfg.report_lag_sigma, size=len(ordered)
    )
    lag_days = np.clip(lag_days, 0.0, float(cfg.maturity_days))
    is_fraud = ordered[schema.TARGET_COL].to_numpy(dtype=int)
    event_dt = ordered[schema.TIME_COL].to_numpy(dtype="int64")
    lag_seconds = (lag_days * SECONDS_PER_DAY).astype("int64")
    delay = np.where(is_fraud == 1, lag_seconds, cfg.maturity_seconds)
    return pd.DataFrame(
        {
            "transaction_id": ordered[schema.ID_COL].to_numpy(dtype="int64"),
            "is_fraud": is_fraud,
            "event_dt": event_dt,
            "observed_dt": event_dt + delay,
        }
    )


def arrived_by(arrivals: pd.DataFrame, as_of_dt: int) -> pd.DataFrame:
    """The outcomes known at `as_of_dt` — what a real feed would have delivered."""
    return arrivals.loc[arrivals["observed_dt"] <= as_of_dt].reset_index(drop=True)


def mature_rows(df: pd.DataFrame, as_of_day: int, maturity_days: int) -> pd.DataFrame:
    """Rows whose reporting window has closed by `as_of_day`: usable as training data.

    Everything younger has a label only if it was reported — a positive-enriched
    sample that must not be trained on.
    """
    day = df[schema.TIME_COL] // SECONDS_PER_DAY
    return df.loc[day <= as_of_day - maturity_days]


def attach_outcomes(
    events: pd.DataFrame, outcomes: pd.DataFrame, as_of_dt: int, maturity_days: int
) -> pd.DataFrame:
    """Join stored outcomes onto prediction events and classify each event's label state.

    Adds `is_fraud` (NaN when unknown), `label_status` and `cohort_closed` (the
    transaction is old enough for a missing report to mean "legitimate").
    """
    known = outcomes[["transaction_id", "is_fraud", "observed_dt"]]
    known = known.drop_duplicates("transaction_id")
    joined = events.merge(known, on="transaction_id", how="left")
    if "transaction_dt" not in joined:
        joined["transaction_dt"] = np.nan
    when = pd.to_numeric(joined["transaction_dt"], errors="raise")
    closed = when + maturity_days * SECONDS_PER_DAY <= as_of_dt
    has_label = joined["is_fraud"].notna()
    status = np.select(
        [has_label, when.isna(), closed],
        ["matured", "unknown_time", "overdue"],
        default="pending",
    )
    joined["label_status"] = pd.Categorical(status, categories=STATUSES)
    joined["cohort_closed"] = closed.fillna(False).astype(bool)
    return joined


def label_section(attached: pd.DataFrame, as_of_dt: int, maturity_days: int) -> dict[str, Any]:
    """Coverage of the window by labels, and the bias a naive reading would carry."""
    n = int(len(attached))
    counts = {s: int((attached["label_status"] == s).sum()) for s in STATUSES}
    closed = attached.loc[attached["cohort_closed"]]
    matured = attached.loc[attached["label_status"] == "matured"]
    positives = matured.loc[matured["is_fraud"] == 1]
    lag_days = (positives["observed_dt"] - positives["transaction_dt"]) / SECONDS_PER_DAY
    when = pd.to_numeric(attached["transaction_dt"], errors="raise")
    pending_age = (as_of_dt - when[attached["label_status"] == "pending"]) / SECONDS_PER_DAY
    return {
        "as_of_day": as_of_dt // SECONDS_PER_DAY,
        "maturity_days": maturity_days,
        "events": n,
        **counts,
        "coverage": counts["matured"] / n if n else None,
        "closed_cohort": int(len(closed)),
        "closed_share": float(len(closed) / n) if n else None,
        # the unbiased rate (closed cohorts only) next to the one a naive join would report
        "positive_rate_closed": float(closed["is_fraud"].mean()) if len(closed) else None,
        "positive_rate_arrived": float(matured["is_fraud"].mean()) if len(matured) else None,
        "report_lag_days_median": float(lag_days.median()) if len(lag_days) else None,
        "report_lag_days_p90": float(lag_days.quantile(0.9)) if len(lag_days) else None,
        "oldest_pending_days": float(pending_age.max()) if len(pending_age) else None,
    }


def early_section(attached: pd.DataFrame) -> dict[str, Any]:
    """What the open cohorts already say: of the fraud reported so far, how much was caught.

    This is biased towards quickly reported fraud and says nothing about false
    positives; it is a leading indicator, not a performance number.
    """
    timed = attached["label_status"] != "unknown_time"
    open_rows = attached.loc[~attached["cohort_closed"] & timed]
    reported = open_rows.loc[open_rows["is_fraud"] == 1]
    n_rep = int(len(reported))
    return {
        "open_cohort": int(len(open_rows)),
        "fraud_reported_so_far": n_rep,
        "reported_rate_so_far": float(n_rep / len(open_rows)) if len(open_rows) else None,
        "early_recall_block": float((reported["action"] == "block").mean()) if n_rep else None,
        "early_recall_block_plus_review": (
            float((reported["action"] != "approve").mean()) if n_rep else None
        ),
    }
