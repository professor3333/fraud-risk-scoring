"""Monitoring report from the prediction audit trail against the frozen reference.

Example:
    uv run python scripts/monitor.py                                   # everything logged so far
    uv run python scripts/monitor.py --since 2026-09-15T16:00 --labels labels.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from fraud.monitor.reference import load_reference
from fraud.monitor.report import build_report, render_markdown
from fraud.serve.audit import AuditLog

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-config", type=Path, default=ROOT / "configs" / "serving.yaml")
    parser.add_argument(
        "--audit-db", type=Path, default=None, help="default: serving.yaml audit_db"
    )
    parser.add_argument("--since", default=None, help="ISO timestamp (UTC), inclusive")
    parser.add_argument("--until", default=None, help="ISO timestamp (UTC), exclusive")
    parser.add_argument("--labels", type=Path, default=None, help="CSV with transaction_id,isFraud")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "monitoring")
    parser.add_argument("--name", default=None, help="report file stem (default: timestamp)")
    args = parser.parse_args()

    serving = yaml.safe_load(args.serving_config.read_text())
    artifact = ROOT / serving["model_path"]
    ref = load_reference(artifact)
    audit = AuditLog(args.audit_db or ROOT / serving["audit_db"])
    requests = audit.frame("requests", args.since, args.until)
    events = audit.frame("prediction_events", args.since, args.until)
    inputs = audit.frame("input_features", args.since, args.until)
    labels = pd.read_csv(args.labels)[["transaction_id", "isFraud"]] if args.labels else None
    report = build_report(requests, events, inputs, ref, labels, (args.since, args.until))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or datetime.now().strftime("%Y%m%dT%H%M%S")
    (args.out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    md = render_markdown(report)
    (args.out_dir / f"{stem}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
