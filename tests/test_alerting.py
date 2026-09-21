"""Scheduled-monitoring tests: what a report means for the operator, and how it is fetched.

The detection thresholds are tested in test_monitoring.py; here the question is the
layer above them (ADR 0011) — a quiet window must never page, an episode must be one
issue rather than one per run, and a refusal from the service must arrive as a readable
error instead of a traceback in a cron log.
"""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fraud.monitor.alerts import AlertConfig, decide, load_alert_config
from fraud.monitor.remote import fetch_report, window_bounds

ROOT = Path(__file__).resolve().parents[1]
CONFIG = AlertConfig(
    window_hours=24, min_rows=200, min_requests=20, alert_on="alert", label="monitoring",
    title="Monitoring alert: live service", timeout_s=180.0,
)  # fmt: skip


def _report(status: str, flags: list[str], rows: int) -> dict[str, Any]:
    """The shape build_report returns, filled in enough for render_markdown."""
    requests = max(rows // 100, 1)
    actions = {"block": 0.02, "review": 0.07, "approve": 0.91}
    return {
        "generated_at": "2026-09-21T07:10:00+00:00",
        "window": {"since": "2026-09-20T07:00:00+00:00", "until": "2026-09-21T07:00:00+00:00"},
        "api": {
            "requests": requests,
            "rows_scored": rows,
            "window_seconds": 86_400.0,
            "requests_per_second": requests / 86_400,
            "latency_p50_ms": 48.0,
            "latency_p95_ms": 120.0,
            "latency_max_ms": 900.0,
            "errors": 0,
            "error_rate": 0.0,
            "by_endpoint": {
                "/predict": {
                    "requests": requests,
                    "rows": rows,
                    "latency_p50_ms": 48.0,
                    "latency_p95_ms": 120.0,
                    "errors": 0,
                }
            },  # fmt: skip
        },
        "predictions": {
            "rows": rows,
            "score": {"mean": 0.036, "reference_mean": 0.036, "p50": 0.01, "p99": 0.5},
            "score_psi": 0.02,
            "score_drift": "ok",
            "actions": actions,
            "reference_actions": actions,
            "action_psi": 0.001,
            "action_drift": "ok",
            "policies": {"rank": requests},
            "model_versions": {"xgb@abc": requests},
        },
        "data": {"rows": rows, "features": {}, "worst_psi": 0.02},
        "model": {
            "reference_performance": {
                "pr_auc": 0.637,
                "roc_auc": 0.93,
                "block_precision": 0.8,
                "cal_brier": 0.0185,
                "cal_ece": 0.0037,
            },  # fmt: skip
            "eventual": None,
        },
        "flags": flags,
        "status": status,
    }


def test_the_shipped_alerting_config_loads_and_is_consistent() -> None:
    config = load_alert_config(ROOT / "configs" / "alerting.yaml")
    assert config.alert_on in ("warn", "alert")
    assert config.window_hours > 0 and config.min_rows > 0 and config.timeout_s > 0
    assert config.min_requests > 0
    assert config.title and config.label


def test_a_quiet_window_is_no_data_and_never_pages() -> None:
    """The public demo is idle for days at a time; an unmeasured window is not an alert."""
    alert = decide(_report("alert", ["score drift alert (PSI 0.31)"], rows=3), CONFIG)
    assert alert.severity == "no_data" and alert.paging is False
    assert alert.flags == ()  # flags from 3 rows say nothing; they are not repeated as fact
    assert "Nothing was measured" in alert.summary


@pytest.mark.parametrize(
    ("status", "alert_on", "paging"),
    [
        ("ok", "alert", False),
        ("warn", "alert", False),  # one PSI over 0.10 is a daily event, not a page
        ("alert", "alert", True),
        ("warn", "warn", True),  # unless the operator asked for warnings
        ("ok", "warn", False),
    ],
)
def test_paging_follows_the_configured_severity(status: str, alert_on: str, paging: bool) -> None:
    config = AlertConfig(**{**CONFIG.__dict__, "alert_on": alert_on})
    alert = decide(_report(status, ["something"] if status != "ok" else [], rows=5_000), config)
    assert alert.severity == status and alert.paging is paging


def test_an_erroring_service_alerts_even_with_nothing_scored() -> None:
    """A service failing every call scores nothing; it must not be read as idle."""
    broken = _report("alert", ["error rate 100.0%"], rows=0)
    broken["api"] = {**broken["api"], "requests": 40, "errors": 40, "error_rate": 1.0}
    alert = decide(broken, CONFIG, source="https://demo")
    assert alert.severity == "alert" and alert.paging is True
    assert alert.flags == ("error rate 100.0%",)
    assert "scored nothing" in alert.summary

    # …but a couple of failed calls on an idle day still say nothing
    quiet = _report("alert", ["error rate 100.0%"], rows=0)
    quiet["api"] = {**quiet["api"], "requests": 2, "errors": 2, "error_rate": 1.0}
    assert decide(quiet, CONFIG).severity == "no_data"


def test_a_label_feed_gap_survives_the_volume_floor() -> None:
    """Labels stopping is true however little was scored in the window."""
    gap = _report("alert", ["label feed gap: 812 transactions past the 120-day window"], rows=10)
    gap["api"] = {**gap["api"], "requests": 30}
    alert = decide(gap, CONFIG)
    assert alert.severity == "alert" and alert.paging and len(alert.flags) == 1


def test_drift_flags_do_not_survive_the_volume_floor() -> None:
    thin = _report("alert", ["score drift alert (PSI 0.31)", "error rate 30.0%"], rows=12)
    thin["api"] = {**thin["api"], "requests": 30, "error_rate": 0.30}
    alert = decide(thin, CONFIG)
    assert alert.flags == ("error rate 30.0%",)  # the PSI over twelve rows is not evidence


def test_an_unknown_status_is_refused_rather_than_downgraded() -> None:
    with pytest.raises(ValueError, match="not one of"):
        decide(_report("degraded", [], rows=5_000), CONFIG)


def test_the_body_carries_the_flags_the_report_and_where_it_came_from() -> None:
    flags = ["eventual PR-AUC 0.590 is -0.047 vs reference", "feature drift alert: amount"]
    alert = decide(
        _report("alert", flags, rows=5_000), CONFIG, source="https://demo", run_url="https://run"
    )
    assert alert.paging and alert.title == CONFIG.title
    for flag in flags:
        assert flag in alert.body
    assert "# Monitoring report — ALERT" in alert.body  # the report itself, not a summary of it
    assert "https://demo" in alert.body and "https://run" in alert.body
    assert "docs/monitoring.md" in alert.body


def test_an_episode_is_one_issue_opened_commented_and_closed() -> None:
    """Three consecutive runs: breach, breach again, recovery."""
    from scripts.alert import deliver

    calls: list[Sequence[str]] = []
    issues: list[dict[str, Any]] = []

    def runner(args: Sequence[str]) -> str:
        calls.append(list(args))
        if args[:2] == ["issue", "list"]:
            return json.dumps(issues)
        if args[:2] == ["issue", "create"]:
            issues.append({"number": 7, "title": args[args.index("--title") + 1]})
            return "https://github.com/o/r/issues/7\n"
        if args[:2] == ["issue", "close"]:
            issues.clear()
        return ""

    breach = decide(_report("alert", ["error rate 9.0%"], rows=5_000), CONFIG)
    assert deliver(breach, CONFIG.label, "o/r", runner) == "opened https://github.com/o/r/issues/7"
    assert deliver(breach, CONFIG.label, "o/r", runner) == "commented on issue #7"

    recovered = decide(_report("ok", [], rows=5_000), CONFIG)
    assert deliver(recovered, CONFIG.label, "o/r", runner) == "closed issue #7"
    assert deliver(recovered, CONFIG.label, "o/r", runner) == "nothing to deliver"
    assert sum(1 for c in calls if c[:2] == ["issue", "create"]) == 1


def test_an_unrelated_labelled_issue_is_left_alone() -> None:
    from scripts.alert import deliver

    def runner(args: Sequence[str]) -> str:
        if args[:2] == ["issue", "list"]:
            return json.dumps([{"number": 3, "title": "monitoring: add a scheduler"}])
        if args[:2] == ["issue", "create"]:
            return "https://github.com/o/r/issues/9\n"
        return ""

    quiet = decide(_report("ok", [], rows=5_000), CONFIG)
    assert deliver(quiet, CONFIG.label, "o/r", runner) == "nothing to deliver"  # not closed
    breach = decide(_report("alert", ["x"], rows=5_000), CONFIG)
    assert "opened" in deliver(breach, CONFIG.label, "o/r", runner)  # not commented on #3


def test_window_bounds_are_utc_and_exactly_the_requested_span() -> None:
    until = datetime(2026, 9, 21, 7, 0, 0, tzinfo=UTC)
    since, end = window_bounds(24, until)
    assert end == "2026-09-21T07:00:00+00:00" and since == "2026-09-20T07:00:00+00:00"


def test_fetch_sends_the_admin_key_and_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _Response:
        def read(self) -> bytes:
            return json.dumps({"status": "ok"}).encode()

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float = 0) -> _Response:
        seen["url"] = request.full_url
        seen["key"] = request.get_header("X-api-key")
        return _Response()

    monkeypatch.setattr("fraud.monitor.remote.urlopen", fake_urlopen)
    report = fetch_report("https://demo/", "2026-09-20T07:00:00+00:00", None, "secret")
    assert report == {"status": "ok"}
    assert seen["url"].startswith("https://demo/audit/monitor?since=2026-09-20T07")
    assert seen["key"] == "secret"


def test_a_refusal_from_the_service_is_a_readable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """403 (admin disabled), 401 (wrong key) and 404 (no reference) are operator actions."""

    def fake_urlopen(request: Any, timeout: float = 0) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr("fraud.monitor.remote.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        fetch_report("https://demo", None, None, None)


def test_an_existing_label_is_not_recreated_or_recoloured() -> None:
    """An operator who recoloured the label meant it; only a missing one is created."""
    from scripts.alert import ensure_label

    calls: list[Sequence[str]] = []

    def runner(args: Sequence[str]) -> str:
        calls.append(list(args))
        return json.dumps([{"name": "monitoring"}]) if args[:2] == ["label", "list"] else ""

    ensure_label("monitoring", "o/r", runner)
    assert [c[:2] for c in calls] == [["label", "list"]]
    ensure_label("other-label", "o/r", runner)
    assert ["label", "create"] in [c[:2] for c in calls]
