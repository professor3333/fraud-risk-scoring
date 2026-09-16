"""Verify a deployed service end to end: health + parity, model-info, and a scored sample.

Standard library only, so it runs on a bare CI runner. When the service requires an
API key (/health reports auth: api_key) it is read from FRAUD_API_KEY or --api-key.

Example:
    uv run python scripts/deploy_check.py https://fraud-risk-scoring.fly.dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
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
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    global API_KEY
    API_KEY = args.api_key

    health = get(f"{base}/health")
    assert health["status"] == "ok", health
    assert health["parity_rows"] > 0, "service started without a parity check"
    if health.get("auth") == "api_key":
        assert API_KEY, "the service requires an API key; pass --api-key or set FRAUD_API_KEY"
    info = get(f"{base}/model-info")
    assert info["version"] == health["model_version"]
    scored = post_csv(f"{base}/predict/csv", SAMPLE)
    s = scored["summary"]
    assert s["analysed"] == 200 and s["model_version"] == health["model_version"]
    top = scored["rows"][0]
    print(f"ok  {base}")
    print(f"    model {info['version']}  parity_rows {health['parity_rows']}")
    print(f"    auth {health.get('auth', 'open')}")
    print(f"    bands {info['bands']}  default budget {info['default_review_budget']}")
    counts = f"block {s['high_risk']}, review {s['review']}, approve {s['approve']}"
    print(f"    sample: analysed {s['analysed']}, {counts}")
    print(f"    top row {top['transaction_id']}: p={top['fraud_probability']:.4f} {top['action']}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        sys.exit(f"deployment check failed: {exc}")
