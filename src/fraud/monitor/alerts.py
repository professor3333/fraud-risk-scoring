"""From a monitoring report to an operator alert (ADR 0011).

`scripts/monitor.py` answers *what is happening*; this module answers *whether
anyone should be woken up*, and it is deliberately the only place that decides.
The report's own ``status`` is not enough on its own for a scheduled job:

- a window with almost no traffic produces an "ok" that means "nothing was
  measured", not "everything is fine", so a volume floor comes first;
- "warn" (a single PSI above 0.10) is a daily occurrence on live traffic and is
  not worth an alert unless the operator asks for it (``alert_on``).

The functions are pure: a report dict in, an :class:`Alert` out. Delivery lives
in `scripts/alert.py`, so the decision can be tested without a network or `gh`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fraud.monitor.report import ERROR_RATE_FLAG, LABEL_GAP_FLAG

#: Ordered by seriousness; ``no_data`` is not a failure, it is an absence of evidence.
SEVERITIES = ("no_data", "ok", "warn", "alert")
#: Flags that hold however few rows the window has: the service is erroring, or labels
#: have stopped arriving. Everything else compares distributions and needs volume.
AVAILABILITY_FLAGS = (ERROR_RATE_FLAG, LABEL_GAP_FLAG)
BODY_LIMIT = 60_000  # GitHub caps an issue body at 65,536 characters


@dataclass(frozen=True)
class AlertConfig:
    """Orchestration knobs for the scheduled monitor (configs/alerting.yaml).

    The *detection* thresholds (PSI 0.10 / 0.20, PR-AUC −0.03, block precision
    −0.05, error rate 5 %) belong to the report and stay in
    `fraud.monitor.report`; what lives here is when to run, when to believe the
    window, and what deserves a notification.
    """

    window_hours: int
    min_rows: int
    min_requests: int  # calls the window needs before an error rate is believed
    alert_on: str  # "warn" or "alert": the lowest severity that notifies
    label: str
    title: str
    timeout_s: float

    def __post_init__(self) -> None:
        if self.alert_on not in ("warn", "alert"):
            raise ValueError(f"alert_on must be 'warn' or 'alert', got {self.alert_on!r}")
        if self.window_hours <= 0 or self.min_rows < 0 or self.min_requests < 1:
            raise ValueError(
                "window_hours and min_requests must be positive and min_rows non-negative"
            )


def load_alert_config(path: Path) -> AlertConfig:
    raw = yaml.safe_load(path.read_text())
    issue = raw.get("issue", {})
    return AlertConfig(
        window_hours=int(raw["window_hours"]),
        min_rows=int(raw["min_rows"]),
        min_requests=int(raw["min_requests"]),
        alert_on=str(raw["alert_on"]),
        label=str(issue["label"]),
        title=str(issue["title"]),
        timeout_s=float(raw.get("timeout_s", 120)),
    )


@dataclass(frozen=True)
class Alert:
    """The verdict on one report: what it was, and whether it notifies."""

    severity: str  # one of SEVERITIES
    paging: bool
    title: str
    summary: str
    body: str
    flags: tuple[str, ...]


def _rows(report: dict[str, Any]) -> int:
    predictions: dict[str, Any] = report.get("predictions", {})
    return int(predictions.get("rows", 0))


def _window(report: dict[str, Any]) -> str:
    w: dict[str, Any] = report.get("window", {})
    return f"{w.get('since') or 'start'} → {w.get('until') or 'now'}"


def summarise(report: dict[str, Any], severity: str, source: str | None = None) -> str:
    """One line an operator can read in a notification without opening anything."""
    api: dict[str, Any] = report.get("api", {})
    where = f" on {source}" if source else ""
    if severity == "no_data":
        return (
            f"No usable traffic{where} in {_window(report)}: {_rows(report)} predictions, "
            f"{api.get('requests', 0)} requests. Nothing was measured."
        )
    if _rows(report) < 1:
        return (
            f"{severity.upper()}{where}: the service scored nothing in {_window(report)} "
            f"over {api.get('requests', 0)} requests, error rate "
            f"{api.get('error_rate', 0.0):.1%}."
        )
    flags = report.get("flags") or []
    head = f"{len(flags)} flag(s)" if flags else "no flags"
    return (
        f"{severity.upper()}{where}: {head} over {_rows(report)} predictions in "
        f"{_window(report)} ({api.get('requests', 0)} requests, "
        f"error rate {api.get('error_rate', 0.0):.1%})."
    )


def render_body(
    report: dict[str, Any], alert: Alert, source: str | None, run_url: str | None
) -> str:
    """The issue body: the summary, then the report itself, then where it came from."""
    from fraud.monitor.report import render_markdown

    lines = [
        alert.summary,
        "",
        *([f"- {f}" for f in alert.flags] or ["- no flags"]),
        "",
        "---",
        "",
        render_markdown(report),
        "",
        f"Source: `{source}`" if source else "Source: local audit database",
    ]
    if run_url:
        lines.append(f"Workflow run: {run_url}")
    lines.append("")
    lines.append(
        "Triage: `docs/monitoring.md` → Scheduled monitoring. A drift flag is not by "
        "itself a reason to retrain (`docs/retraining.md`); an eventual-metric flag is."
    )
    body = "\n".join(lines)
    return body if len(body) <= BODY_LIMIT else body[:BODY_LIMIT] + "\n\n…truncated.\n"


def decide(
    report: dict[str, Any],
    config: AlertConfig,
    source: str | None = None,
    run_url: str | None = None,
) -> Alert:
    """The severity of one report and whether it notifies.

    A window below ``min_rows`` predictions is ``no_data``: its PSIs are computed
    from too few rows to mean anything, so it is recorded and never paged on. The
    exception is the availability flags — a service erroring on every call also
    scores nothing, and a silent "no data" is the wrong answer to an outage.
    """
    flags = tuple(str(f) for f in report.get("flags", ()))
    status = str(report.get("status", "ok"))
    if status not in SEVERITIES:
        raise ValueError(f"report status {status!r} is not one of {SEVERITIES}")
    if _rows(report) >= config.min_rows:
        severity = status
    else:
        # The floor suppresses distribution comparisons, which need rows to mean
        # anything — not availability, which is true however few rows there are. A
        # service erroring on every call scores nothing, and must not look idle.
        flags = tuple(f for f in flags if f.startswith(AVAILABILITY_FLAGS))
        api: dict[str, Any] = report.get("api", {})
        severity = (
            "alert" if flags and int(api.get("requests", 0)) >= config.min_requests else "no_data"
        )
        if severity == "no_data":
            flags = ()
    paging = severity != "no_data" and (
        SEVERITIES.index(severity) >= SEVERITIES.index(config.alert_on)
    )
    summary = summarise(report, severity, source)
    alert = Alert(
        severity=severity,
        paging=paging,
        title=config.title,
        summary=summary,
        body="",
        flags=flags,
    )
    return Alert(**{**alert.__dict__, "body": render_body(report, alert, source, run_url)})
