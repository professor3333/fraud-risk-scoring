"""The label feed, simulated: append the outcomes that have arrived by a given day.

Fraud outcomes arrive after the transaction (docs/feedback.md). The dataset has the
labels but not the dates they became known, so this script draws those dates
(configs/feedback.yaml) and plays the feed forward: every run advances the clock to
--as-of-day and appends the outcomes known by then for the transactions the service
has scored. It is idempotent — rerunning at the same day appends nothing.

Example:
    uv run python scripts/feedback.py --as-of-day 160
    uv run python scripts/feedback.py --as-of-day 200 --all   # every transaction, not only scored
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fraud.data import schema
from fraud.data.load import load_train
from fraud.monitor.feedback import (
    SECONDS_PER_DAY,
    arrived_by,
    attach_outcomes,
    label_section,
    load_feedback_config,
    simulate_arrivals,
)
from fraud.serve.audit import AuditLog, utc_now

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-day", type=int, required=True, help="advance the feed to this day")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "feedback.yaml")
    parser.add_argument("--serving-config", type=Path, default=ROOT / "configs" / "serving.yaml")
    parser.add_argument(
        "--audit-db", type=Path, default=None, help="default: serving.yaml audit_db"
    )
    parser.add_argument("--all", action="store_true", help="feed every transaction's outcome")
    args = parser.parse_args()

    cfg = load_feedback_config(args.config)
    serving = yaml.safe_load(args.serving_config.read_text())
    audit = AuditLog(args.audit_db or ROOT / serving["audit_db"])
    clock = audit.feed_clock()
    as_of_dt = (args.as_of_day + 1) * SECONDS_PER_DAY - 1  # end of the day
    if clock is not None and as_of_dt < clock:
        raise SystemExit(f"the feed is at day {clock // SECONDS_PER_DAY}; it does not rewind")

    df = load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed")
    arrivals = simulate_arrivals(df[[schema.ID_COL, schema.TIME_COL, schema.TARGET_COL]], cfg)
    known = arrived_by(arrivals, as_of_dt)
    events = audit.frame("prediction_events")
    if not args.all:
        known = known.loc[known["transaction_id"].isin(events["transaction_id"])]
    now = utc_now()
    rows = [{**r, "recorded_at": now, "source": "simulated_feed"} for r in known.to_dict("records")]
    n_new = audit.record_outcomes(rows)
    total = audit.record_feed_run(as_of_dt, n_new, "simulated_feed")
    print(f"feed at day {args.as_of_day}: +{n_new} outcomes ({total} stored)")
    if events.empty:
        print("no predictions stored yet; nothing to attach to")
        return
    attached = attach_outcomes(events, audit.frame("outcomes"), as_of_dt, cfg.maturity_days)
    lab = label_section(attached, as_of_dt, cfg.maturity_days)
    print(
        f"scored transactions {lab['events']}: matured {lab['matured']} · pending {lab['pending']}"
        f" · overdue {lab['overdue']} · closed cohort {lab['closed_cohort']}"
    )
    if lab["positive_rate_arrived"] is not None:
        closed = lab["positive_rate_closed"]
        print(
            f"positive rate among arrived labels {lab['positive_rate_arrived']:.2%}; closed cohort"
            + (f" {closed:.2%}" if closed is not None else " not yet available")
        )


if __name__ == "__main__":
    main()
