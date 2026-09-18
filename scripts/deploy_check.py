"""Verify a deployed service end to end: health + parity, model-info, a scored sample, and
that the admin endpoints (/outcomes, /audit/*) refuse anonymous callers.

Standard library only, so it runs on a bare CI runner. When the service requires an
API key (/health reports auth: api_key) it is read from FRAUD_API_KEY or --api-key.

With --expected-artifact-sha the check stops being only a consistency check and becomes
the deployment contract: the artifact the service has loaded must be the champion this
release was cut for. Without it a stale champion passes everything here, because every
answer is internally consistent — it is consistently the *old* model (docs/promotion.md).

Example:
    uv run python scripts/deploy_check.py https://fraud-risk-scoring.fly.dev
    uv run python scripts/deploy_check.py https://… --expected-artifact-sha 7af85ec92813
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "src" / "fraud" / "serve" / "static" / "sample_transactions.csv"
API_KEY: str | None = None


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"accept": "application/json", **(extra or {})}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def get(url: str) -> dict:
    with urlopen(Request(url, headers=_headers()), timeout=60) as r:
        return json.load(r)


def status_without_key(url: str) -> int:
    """The HTTP status an anonymous GET receives (no X-API-Key at all)."""
    try:
        with urlopen(Request(url, headers={"accept": "application/json"}), timeout=60) as r:
            return int(r.status)
    except HTTPError as e:
        return e.code


def post_csv(url: str, path: Path) -> dict:
    boundary = "----fraudcheck"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{path.name}"\r\nContent-Type: text/csv\r\n\r\n'
        ).encode()
        + path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = Request(
        url,
        data=body,
        method="POST",
        headers=_headers({"Content-Type": f"multipart/form-data; boundary={boundary}"}),
    )
    with urlopen(req, timeout=120) as r:
        return json.load(r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--api-key", default=os.environ.get("FRAUD_API_KEY") or None)
    parser.add_argument(
        "--expected-artifact-sha",
        default=os.environ.get("FRAUD_EXPECTED_ARTIFACT_SHA") or None,
        help="require the live service to have loaded this artifact (a sha256 prefix)",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    expected_sha: str | None = args.expected_artifact_sha
    global API_KEY
    API_KEY = args.api_key

    health = get(f"{base}/health")
    assert health["status"] == "ok", health
    assert health["parity_rows"] > 0, "service started without a parity check"
    if health.get("auth") == "api_key":
        assert API_KEY, "the service requires an API key; pass --api-key or set FRAUD_API_KEY"
    admin = health.get("admin", "disabled")
    refused = status_without_key(f"{base}/audit/recent?limit=1")
    assert refused == {"disabled": 403, "api_key": 401}[admin], (
        f"anonymous /audit/recent answered {refused}; the audit trail and /outcomes must not "
        "be open on a public URL"
    )
    info = get(f"{base}/model-info")
    assert info["version"] == health["model_version"]
    # A service older than this field answers without it; that is only fatal when the
    # contract is actually being asserted, never a reason to crash a plain check.
    served_sha = info.get("artifact_sha256")
    if expected_sha is not None:
        want = expected_sha.strip().lower()
        if served_sha is None:
            sys.exit(
                f"the service at {base} does not report artifact_sha256, so the champion it "
                "has loaded cannot be verified; deploy a build that reports it."
            )
        if not served_sha.lower().startswith(want):
            sys.exit(
                f"the service has loaded artifact {served_sha[:12]}, not the expected {want}. "
                "The host is serving a different champion than this release was cut for; "
                "check FRAUD_CHAMPION_URL on it (docs/promotion.md)."
            )
    scored = post_csv(f"{base}/predict/csv", SAMPLE)
    s = scored["summary"]
    assert s["analysed"] == 200 and s["model_version"] == health["model_version"]
    top = scored["rows"][0]
    print(f"ok  {base}")
    print(f"    model {info['version']}  parity_rows {health['parity_rows']}")
    match_note = "  (matches the expected champion)" if expected_sha else ""
    print(f"    artifact {served_sha[:12] if served_sha else 'not reported'}{match_note}")
    print(f"    auth {health.get('auth', 'open')}  admin {admin} (anonymous /audit → {refused})")
    print(f"    bands {info['bands']}  default budget {info['default_review_budget']}")
    counts = f"block {s['high_risk']}, review {s['review']}, approve {s['approve']}"
    print(f"    sample: analysed {s['analysed']}, {counts}")
    print(f"    top row {top['transaction_id']}: p={top['fraud_probability']:.4f} {top['action']}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        sys.exit(f"deployment check failed: {exc}")
