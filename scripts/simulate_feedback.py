"""Play the delayed-label loop forward on chronological data (docs/feedback.md).

1. Score a window of days with the served artifact, exactly as /predict/csv does
   (same model object, same policy), into a dedicated audit database.
2. Advance the label feed to each clock in turn (scripts/feedback.py's logic) and
   run the monitor at that clock.
3. Write the timeline: what the monitor could see at each point — labels matured
   and pending, the biased and unbiased positive rates, the early signal, and the
   eventual performance once cohorts close.

Nothing after the scored window is scored; the clocks only let time pass so that
outcomes arrive. The eventual numbers use the dataset's labels; only their arrival
dates are simulated.

Example:
    uv run python scripts/simulate_feedback.py                          # days 123-152, then wait
    uv run python scripts/simulate_feedback.py --clocks 152 182 212 272 --fresh
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fraud.data import schema
from fraud.data.load import load_train
from fraud.monitor.feedback import (
    SECONDS_PER_DAY,
    arrived_by,
    load_feedback_config,
    simulate_arrivals,
)
from fraud.monitor.reference import load_reference
from fraud.monitor.report import build_report, render_markdown
from fraud.serve.app import assign_actions, load_state, record_events
from fraud.serve.audit import new_request_id, utc_now

ROOT = Path(__file__).resolve().parents[1]


def score_window(state: object, df: pd.DataFrame, first_day: int, last_day: int) -> int:
    """One request per day through the service's own scoring and audit path."""
    from fraud.serve.app import LEVEL_FOR_ACTION, ServingState

    assert isinstance(state, ServingState)
    day = df[schema.TIME_COL] // SECONDS_PER_DAY
    n = 0
    for d in range(first_day, last_day + 1):
        frame = df.loc[day == d].reset_index(drop=True)
        if frame.empty:
            continue
        t0 = time.perf_counter()
        p = np.asarray(state.model.predict_proba(frame)[:, 1], dtype=float)
        actions, applied = assign_actions(state, p, "threshold", None)
        rows = [
            (int(t), float(pp), LEVEL_FOR_ACTION[a], a)
            for t, pp, a in zip(frame[schema.ID_COL], p, actions, strict=True)
        ]
        latency = (time.perf_counter() - t0) * 1000
        rid = new_request_id()
        record_events(state, "/predict/csv", rid, applied, rows, latency, frame)
        assert state.audit is not None
        state.audit.record_request(rid, "/predict/csv", utc_now(), 200, latency, len(rows))
        n += len(rows)
        print(f"day {d}: scored {len(rows)} rows in {latency / 1000:.1f} s", flush=True)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-config", type=Path, default=ROOT / "configs" / "serving.yaml")
    parser.add_argument("--feedback-config", type=Path, default=ROOT / "configs" / "feedback.yaml")
    parser.add_argument("--days", type=int, nargs=2, default=(123, 152), metavar=("FIRST", "LAST"))
    parser.add_argument(
        "--clocks", type=int, nargs="+", default=(152, 160, 182, 212, 242, 272, 300),
        help="days the feed advances to; each gets a monitoring report",
    )  # fmt: skip
    parser.add_argument(
        "--audit-db", type=Path, default=ROOT / "models" / "audit" / "feedback_sim.sqlite"
    )
    parser.add_argument("--fresh", action="store_true", help="delete an existing simulation db")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "feedback")
    args = parser.parse_args()

    if args.audit_db.exists():
        if not args.fresh:
            raise SystemExit(f"{args.audit_db} exists; pass --fresh to start over")
        for suffix in ("", "-wal", "-shm"):
            Path(str(args.audit_db) + suffix).unlink(missing_ok=True)
    os.environ["FRAUD_AUDIT_DB"] = str(args.audit_db)
    state = load_state(args.serving_config)
    assert state.audit is not None
    fb = load_feedback_config(args.feedback_config)
    serving = yaml.safe_load(args.serving_config.read_text())
    ref = load_reference(ROOT / serving["model_path"])

    df = load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed")
    first, last = args.days
    if last > int(df[schema.TIME_COL].max() // SECONDS_PER_DAY):
        raise SystemExit("the scored window must lie inside the labelled data")
    n_scored = score_window(state, df, first, last)
    print(f"scored {n_scored} transactions on days {first}-{last}")

    arrivals = simulate_arrivals(df[[schema.ID_COL, schema.TIME_COL, schema.TARGET_COL]], fb)
    events = state.audit.frame("prediction_events")
    scored = arrivals.loc[arrivals["transaction_id"].isin(events["transaction_id"])]
    total_fraud = int(scored["is_fraud"].sum())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timeline: list[dict[str, object]] = []
    for clock_day in sorted(args.clocks):
        if clock_day < last:
            raise SystemExit(f"clock {clock_day} is before the end of the scored window")
        clock = (clock_day + 1) * SECONDS_PER_DAY - 1
        known = arrived_by(scored, clock).assign(recorded_at=utc_now(), source="simulated_feed")
        n_new = state.audit.record_outcomes(known.to_dict("records"))
        state.audit.record_feed_run(clock, n_new, "simulated_feed")
        report = build_report(
            state.audit.frame("requests"),
            events,
            state.audit.frame("input_features"),
            ref,
            state.audit.frame("outcomes"),
            clock=clock,
            maturity_days=fb.maturity_days,
        )
        (args.out_dir / f"day_{clock_day}.md").write_text(render_markdown(report))
        (args.out_dir / f"day_{clock_day}.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
        model = report["model"]
        lab, early, ev = model["labels"], model["early"], model["eventual"]
        reported = int((known["is_fraud"] == 1).sum())
        timeline.append(
            {
                "as_of_day": clock_day,
                "days_since_window_end": clock_day - last,
                "outcomes_new": n_new,
                "matured": lab["matured"],
                "pending": lab["pending"],
                "overdue": lab["overdue"],
                "closed_share": lab["closed_share"],
                "fraud_reported_share": reported / total_fraud if total_fraud else None,
                "positive_rate_arrived": lab["positive_rate_arrived"],
                "positive_rate_closed": lab["positive_rate_closed"],
                "early_recall_block": early["early_recall_block"],
                "early_recall_block_plus_review": early["early_recall_block_plus_review"],
                "eventual_n": ev.get("n_labelled"),
                "eventual_pr_auc": ev.get("pr_auc"),
                "eventual_block_precision": ev.get("block_precision"),
                "eventual_recall_block": ev.get("recall_block"),
                "status": report["status"],
                "flags": "; ".join(report["flags"]),
            }
        )
        print(
            f"clock day {clock_day}: +{n_new} outcomes, matured {lab['matured']}, pending"
            f" {lab['pending']}, closed {lab['closed_share']:.0%}, status {report['status']}",
            flush=True,
        )
    table = pd.DataFrame(timeline)
    table.to_csv(args.out_dir / "timeline.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.drop(columns=["flags"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
