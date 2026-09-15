"""Verify a deployed service end to end: health + parity, model-info, and a scored sample.

Example:
    uv run python scripts/deploy_check.py https://fraud-risk-scoring.fly.dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "src" / "fraud" / "serve" / "static" / "sample_transactions.csv"


def get(url: str) -> dict:
    with urlopen(Request(url, headers={"accept": "application/json"}), timeout=60) as r:
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
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urlopen(req, timeout=120) as r:
        return json.load(r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = get(f"{base}/health")
    assert health["status"] == "ok", health
    assert health["parity_rows"] > 0, "service started without a parity check"
    info = get(f"{base}/model-info")
    assert info["version"] == health["model_version"]
    scored = post_csv(f"{base}/predict/csv", SAMPLE)
    s = scored["summary"]
    assert s["analysed"] == 200 and s["model_version"] == health["model_version"]
    top = scored["rows"][0]
    print(f"ok  {base}")
    print(f"    model {info['version']}  parity_rows {health['parity_rows']}")
    print(f"    threshold {info['threshold']}  bands {info['bands']}")
    counts = f"block {s['high_risk']}, review {s['review']}, approve {s['approve']}"
    print(f"    sample: analysed {s['analysed']}, {counts}")
    print(f"    top row {top['transaction_id']}: p={top['fraud_probability']:.4f} {top['action']}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        sys.exit(f"deployment check failed: {exc}")
