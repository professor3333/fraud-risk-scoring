"""Monitoring report from the prediction audit trail against the frozen reference.

Two sources, one report. Locally the audit database is a file this process can
open; for a deployed service it is inside the container, so the report is built
there and fetched over the admin endpoint (ADR 0011).

Example:
    uv run python scripts/monitor.py                                   # everything logged so far
    uv run python scripts/monitor.py --since 2026-09-15T16:00 --labels labels.csv
    uv run python scripts/monitor.py --from-url https://… --window-hours 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fraud.monitor.feedback import SECONDS_PER_DAY, load_feedback_config
from fraud.monitor.reference import load_reference
from fraud.monitor.remote import fetch_report, window_bounds
from fraud.monitor.report import build_report, render_markdown
from fraud.serve.audit import AuditLog

ROOT = Path(__file__).resolve().parents[1]
SEVERITY_ORDER = ("ok", "warn", "alert")


def local_report(args: argparse.Namespace, since: str | None, until: str | None) -> dict[str, Any]:
    """Build the report here, from an audit database this process can open."""
    serving = yaml.safe_load(args.serving_config.read_text())
    artifact = ROOT / serving["model_path"]
    ref = load_reference(artifact)
    audit = AuditLog(args.audit_db or ROOT / serving["audit_db"])
    requests = audit.frame("requests", since, until)
    events = audit.frame("prediction_events", since, until)
    inputs = audit.frame("input_features", since, until)
    if args.labels:
        outcomes = pd.read_csv(args.labels)[["transaction_id", "isFraud"]].rename(
            columns={"isFraud": "is_fraud"}
        )
        clock, maturity = None, 0
    else:
        fb = load_feedback_config(args.feedback_config)
        maturity = fb.maturity_days
        clock = audit.feed_clock()
        if args.as_of_day is not None:
            clock = (args.as_of_day + 1) * SECONDS_PER_DAY - 1
        outcomes = audit.frame("outcomes") if clock is not None else None
    return build_report(requests, events, inputs, ref, outcomes, (since, until), clock, maturity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-config", type=Path, default=ROOT / "configs" / "serving.yaml")
    parser.add_argument(
        "--audit-db", type=Path, default=None, help="default: serving.yaml audit_db"
    )
    parser.add_argument("--since", default=None, help="ISO timestamp (UTC), inclusive")
    parser.add_argument("--until", default=None, help="ISO timestamp (UTC), exclusive")
    parser.add_argument(
        "--window-hours", type=float, default=None,
        help="trailing window ending now; an alternative to --since/--until",
    )  # fmt: skip
    parser.add_argument(
        "--from-url", default=None,
        help="base URL of a running service; fetch GET /audit/monitor instead of reading a"
        " local database. The admin key comes from FRAUD_ADMIN_API_KEY or --admin-key.",
    )  # fmt: skip
    parser.add_argument("--admin-key", default=os.environ.get("FRAUD_ADMIN_API_KEY") or None)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds, with --from-url")
    parser.add_argument(
        "--labels", type=Path, default=None,
        help="CSV with transaction_id,isFraud taken as complete and final; default: the outcomes"
        " table, read as of the label feed's clock",
    )  # fmt: skip
    parser.add_argument("--feedback-config", type=Path, default=ROOT / "configs" / "feedback.yaml")
    parser.add_argument(
        "--as-of-day", type=int, default=None, help="override the feed clock (TransactionDT day)"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "monitoring")
    parser.add_argument("--name", default=None, help="report file stem (default: timestamp)")
    parser.add_argument(
        "--fail-on", choices=("never", "warn", "alert"), default="never",
        help="exit 1 when the report is at least this severe (default: never; the scheduled"
        " job leaves the verdict to scripts/alert.py)",
    )  # fmt: skip
    args = parser.parse_args()

    if args.window_hours is not None:
        if args.since or args.until:
            parser.error("--window-hours cannot be combined with --since/--until")
        since, until = window_bounds(args.window_hours)
    else:
        since, until = args.since, args.until

    if args.from_url:
        if args.labels or args.as_of_day is not None:
            parser.error("--labels and --as-of-day apply to a local audit database only")
        report = fetch_report(args.from_url, since, until, args.admin_key, args.timeout)
    else:
        report = local_report(args, since, until)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or datetime.now().strftime("%Y%m%dT%H%M%S")
    (args.out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    md = render_markdown(report)
    (args.out_dir / f"{stem}.md").write_text(md)
    print(md)

    if args.fail_on != "never":
        status = str(report["status"])
        if SEVERITY_ORDER.index(status) >= SEVERITY_ORDER.index(args.fail_on):
            print(
                f"monitoring status {status} at or above --fail-on {args.fail_on}", file=sys.stderr
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
