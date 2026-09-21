"""Fetching a monitoring report from a running service.

The audit trail lives inside the service (a SQLite file in its container), so a
scheduler outside it cannot open the database. Two ways out: ship the rows to
the scheduler and compute there, or ask the service for the report it can
already build. This module does the second (ADR 0011) — the same
`fraud.monitor.report.build_report` runs in both places, and no prediction rows
leave the service.

Standard library only: the scheduler is a bare CI runner, like
`scripts/deploy_check.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def window_bounds(window_hours: float, until: datetime | None = None) -> tuple[str, str]:
    """The ISO-8601 UTC bounds of the trailing window, as the audit trail stores them."""
    end = (until or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    start = end - timedelta(hours=window_hours)
    return start.isoformat(), end.isoformat()


def fetch_report(
    base_url: str,
    since: str | None,
    until: str | None,
    admin_key: str | None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """GET /audit/monitor from a deployed service.

    Raises ``RuntimeError`` with the service's own message on a refusal, because
    every likely failure (403 admin disabled, 401 wrong key, 404 no reference
    frozen for the served artifact) is an operator action, not a bug.
    """
    query = {k: v for k, v in (("since", since), ("until", until)) if v}
    url = f"{base_url.rstrip('/')}/audit/monitor" + (f"?{urlencode(query)}" if query else "")
    headers = {"accept": "application/json"}
    if admin_key:
        headers["X-API-Key"] = admin_key
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            body: dict[str, Any] = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"GET {url} failed: HTTP {exc.code} {detail}") from exc
    return body
