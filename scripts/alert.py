"""Deliver a monitoring report's verdict to the operator as a GitHub issue (ADR 0011).

The scheduled job runs `scripts/monitor.py` and then this. What it does:

    paging + no open alert   -> open one, labelled, with the report in the body
    paging + an open alert   -> comment on it (one issue per episode, not per day)
    quiet  + an open alert   -> comment "recovered" and close it
    quiet  + no open alert   -> print the summary and stop

So a week of drift is one issue with seven comments, and the issue closing is
itself the signal that the service came back. Delivery is `gh`, because the
repository already authenticates with it and a webhook would be one more
secret to hold for a project with one operator.

Example:
    uv run python scripts/alert.py reports/monitoring/20260921T070000.json \
        --source https://fraud-risk-scoring-m1fp.onrender.com --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from fraud.monitor.alerts import Alert, decide, load_alert_config

ROOT = Path(__file__).resolve().parents[1]
Runner = Callable[[Sequence[str]], str]


def gh(args: Sequence[str]) -> str:
    """Run `gh` and return stdout; a failure is loud, with the command in the message."""
    done = subprocess.run(["gh", *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def open_alert_issue(label: str, title: str, repo: str | None, runner: Runner) -> int | None:
    """The number of the open alert issue, or None. Matched on label *and* title, so an
    unrelated issue a human labelled by hand is never commented on or closed."""
    args = ["issue", "list", "--label", label, "--state", "open", "--json", "number,title",
            "--limit", "20"]  # fmt: skip
    if repo:
        args += ["--repo", repo]
    issues: list[dict[str, Any]] = json.loads(runner(args) or "[]")
    for issue in issues:
        if issue.get("title") == title:
            return int(issue["number"])
    return None


def ensure_label(label: str, repo: str | None, runner: Runner) -> None:
    """Create the alert label if the repository has none. Never `--force`: an operator who
    recoloured or redescribed it meant it."""
    repo_args = ["--repo", repo] if repo else []
    listed = runner(["label", "list", "--json", "name", "--limit", "100", *repo_args])
    if label not in {row["name"] for row in json.loads(listed or "[]")}:
        runner(["label", "create", label, "--color", "B60205",
                "--description", "Automated monitoring alert (docs/monitoring.md)",
                *repo_args])  # fmt: skip


def deliver(alert: Alert, label: str, repo: str | None, runner: Runner) -> str:
    """Create, comment on or close the alert issue. Returns what was done, for the log."""
    repo_args = ["--repo", repo] if repo else []
    number = open_alert_issue(label, alert.title, repo, runner)
    if alert.paging:
        if number is None:
            ensure_label(label, repo, runner)
            url = runner(["issue", "create", "--title", alert.title, "--body", alert.body,
                          "--label", label, *repo_args]).strip()  # fmt: skip
            return f"opened {url}"
        runner(["issue", "comment", str(number), "--body", alert.body, *repo_args])
        return f"commented on issue #{number}"
    if number is not None:
        runner(["issue", "close", str(number), "--comment",
                f"Recovered: {alert.summary}", *repo_args])  # fmt: skip
        return f"closed issue #{number}"
    return "nothing to deliver"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="the JSON report scripts/monitor.py wrote")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "alerting.yaml")
    parser.add_argument("--source", default=None, help="the service the report came from")
    parser.add_argument("--run-url", default=None, help="the workflow run, linked in the body")
    parser.add_argument("--repo", default=None, help="owner/name; default: the checkout's remote")
    parser.add_argument("--dry-run", action="store_true", help="decide and print; touch no issue")
    args = parser.parse_args()

    config = load_alert_config(args.config)
    report = json.loads(args.report.read_text())
    alert = decide(report, config, args.source, args.run_url)

    print(f"severity={alert.severity} paging={alert.paging}")
    print(alert.summary)
    for flag in alert.flags:
        print(f"  - {flag}")

    if args.dry_run:
        print("(dry run: no issue was created, commented on or closed)")
    else:
        print(deliver(alert, config.label, args.repo, gh))

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"severity={alert.severity}\npaging={str(alert.paging).lower()}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(alert.body + "\n")


if __name__ == "__main__":
    main()
