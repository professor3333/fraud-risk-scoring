"""Benchmark the inference service: cold start, latency, throughput, memory.

Uses the 200 synthetic rows shipped with the dashboard (no data download), tiled
to the 1,000-row batch and 5,000-row CSV limits. Without --url it starts a local
uvicorn on a scratch audit database, measures cold start (process launch to a
healthy /health) and samples the server's resident memory; with --url it
benchmarks a running server (a container, the Fly deployment) and reports what
it can observe from outside.

Example:
    uv run python scripts/benchmark_api.py                                 # local uvicorn
    uv run python scripts/benchmark_api.py --url http://127.0.0.1:8000 --name docker
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from fraud.data import schema
from fraud.serve.frames import row_to_payload, table_to_frame

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "src" / "fraud" / "serve" / "static" / "sample_transactions.csv"


# --- HTTP without dependencies ----------------------------------------------------------


def _request(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> float:
    """Latency in ms of one request; raises on a non-2xx status."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000


def post_json(url: str, body: Any) -> float:
    return _request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})


def post_csv(url: str, csv_text: str) -> float:
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="b.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n{csv_text}\r\n--{boundary}--\r\n"
    ).encode()
    return _request(url, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=2) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


# --- the local server, when we own it --------------------------------------------------


class LocalServer:
    def __init__(self, port: int) -> None:
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()

    def start(self) -> float:
        env = {
            **os.environ,
            "FRAUD_AUDIT_DB": str(Path(tempfile.mkdtemp()) / "benchmark_audit.sqlite"),
        }
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "fraud.serve.app:app", "--port", str(self.port),
             "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        while not healthy(self.base):
            if self.proc.poll() is not None:
                raise RuntimeError("the server exited before becoming healthy")
            time.sleep(0.1)
        threading.Thread(target=self._sample_rss, daemon=True).start()
        return (time.perf_counter() - t0) * 1000

    def _sample_rss(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(self.proc.pid)], capture_output=True, text=True
            )
            if out.stdout.strip():
                self.peak_rss_mb = max(self.peak_rss_mb, int(out.stdout.strip()) / 1024)
            time.sleep(0.2)

    def stop(self) -> None:
        self._stop.set()
        self.proc.terminate()
        self.proc.wait(timeout=10)


# --- payloads ---------------------------------------------------------------------------


def load_payloads(n: int) -> list[dict[str, Any]]:
    """The sample rows tiled to n, with unique TransactionIDs."""
    table = pd.read_csv(SAMPLE, dtype="str", keep_default_na=False)
    frame, _ = table_to_frame(table)
    base = [row_to_payload(row) for _, row in frame.iterrows()]
    out = []
    for i in range(n):
        p = dict(base[i % len(base)])
        p[schema.ID_COL] = 10_000_000 + i
        out.append(p)
    return out


def csv_of(payloads: list[dict[str, Any]]) -> str:
    return pd.DataFrame(payloads).to_csv(index=False)


def summary(ms: list[float]) -> dict[str, float]:
    s = sorted(ms)
    return {
        "n": len(s),
        "p50_ms": statistics.median(s),
        "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))],
        "max_ms": s[-1],
        "mean_ms": statistics.fmean(s),
    }


# --- scenarios --------------------------------------------------------------------------


def run(base: str, singles: int, concurrency: int, repeats: int) -> dict[str, Any]:
    payloads = load_payloads(5_000)
    predict, batch, csv = f"{base}/predict", f"{base}/predict/batch", f"{base}/predict/csv"
    for p in payloads[:5]:  # warm-up (first call pays lazy imports and caches)
        post_json(predict, p)
    results: dict[str, Any] = {}

    singles_ms = [post_json(predict, payloads[i % 200]) for i in range(singles)]
    results["single_predict"] = summary(singles_ms)

    results["batch_1000"] = summary(
        [post_json(batch, {"transactions": payloads[:1000], "review_budget": 200})
         for _ in range(repeats)]
    )  # fmt: skip
    for n in (200, 1000, 5000):
        text = csv_of(payloads[:n])
        results[f"csv_{n}"] = summary(
            [post_csv(f"{csv}?review_budget=200", text) for _ in range(repeats)]
        )
        results[f"csv_{n}"]["bytes"] = len(text.encode())

    # concurrent single predictions: the API-integration pattern
    per_worker = max(singles // concurrency, 10)
    errors = 0
    lat: list[float] = []
    lock = threading.Lock()

    def worker(k: int) -> None:
        nonlocal errors
        for i in range(per_worker):
            try:
                ms = post_json(predict, payloads[(k * per_worker + i) % 200])
            except urllib.error.HTTPError:
                with lock:
                    errors += 1
                continue
            with lock:
                lat.append(ms)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(concurrency)))
    wall = time.perf_counter() - t0
    results["concurrent_single"] = {
        **summary(lat),
        "workers": concurrency,
        "requests": concurrency * per_worker,
        "errors": errors,
        "throughput_rps": len(lat) / wall,
        "wall_s": wall,
    }

    # concurrent analyst uploads: several 200-row CSVs at once
    text = csv_of(payloads[:200])
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        csv_lat = list(
            pool.map(lambda _: post_csv(f"{csv}?review_budget=200", text), range(concurrency))
        )
    results["concurrent_csv_200"] = {
        **summary(csv_lat),
        "workers": concurrency,
        "rows_per_s": 200 * concurrency / (time.perf_counter() - t0),
    }
    return results


def render(report: dict[str, Any]) -> str:
    r = report["results"]
    rows = [
        ("cold start (launch → healthy /health)", report.get("cold_start_ms"), None),
        (
            "single /predict, sequential",
            r["single_predict"]["p50_ms"],
            r["single_predict"]["p95_ms"],
        ),
        ("/predict/batch, 1,000 rows", r["batch_1000"]["p50_ms"], r["batch_1000"]["max_ms"]),
        ("/predict/csv, 200 rows", r["csv_200"]["p50_ms"], r["csv_200"]["max_ms"]),
        ("/predict/csv, 1,000 rows", r["csv_1000"]["p50_ms"], r["csv_1000"]["max_ms"]),
        ("/predict/csv, 5,000 rows", r["csv_5000"]["p50_ms"], r["csv_5000"]["max_ms"]),
        (
            f"{r['concurrent_single']['workers']} concurrent clients, single /predict",
            r["concurrent_single"]["p50_ms"],
            r["concurrent_single"]["p95_ms"],
        ),
        (
            f"{r['concurrent_csv_200']['workers']} concurrent 200-row CSV uploads",
            r["concurrent_csv_200"]["p50_ms"],
            r["concurrent_csv_200"]["max_ms"],
        ),
    ]

    def fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:,.0f} ms"

    lines = [
        f"# API benchmark — {report['name']}",
        "",
        f"Target {report['target']}; {report['when']}; {report['machine']}.",
        "",
        "| scenario | p50 | p95 / max |",
        "|---|---:|---:|",
        *[f"| {name} | {fmt(a)} | {fmt(b)} |" for name, a, b in rows],
        "",
        f"- throughput, {r['concurrent_single']['workers']} concurrent single predictions:"
        f" **{r['concurrent_single']['throughput_rps']:.0f} req/s**"
        f" ({r['concurrent_single']['errors']} errors of {r['concurrent_single']['requests']})",
        f"- concurrent CSV: {r['concurrent_csv_200']['rows_per_s']:,.0f} rows/s",
        f"- 5,000-row CSV body: {r['csv_5000']['bytes'] / 1e6:.1f} MB",
    ]
    if report.get("peak_rss_mb") is not None:
        lines.append(
            f"- peak resident memory of the server process: **{report['peak_rss_mb']:.0f} MB**"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="benchmark a running server instead")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--singles", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--name", default="local")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "benchmark")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "name": args.name,
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "machine": f"{os.uname().sysname} {os.uname().machine}, {os.cpu_count()} CPUs",
        "singles": args.singles,
        "concurrency": args.concurrency,
    }
    server: LocalServer | None = None
    if args.url:
        base = args.url.rstrip("/")
        report["target"] = base
        if not healthy(base):
            raise SystemExit(f"{base}/health is not ok")
    else:
        server = LocalServer(args.port)
        report["cold_start_ms"] = server.start()
        base = server.base
        report["target"] = "local uvicorn (one worker)"
    try:
        report["results"] = run(base, args.singles, args.concurrency, args.repeats)
    finally:
        if server is not None:
            server.stop()
            report["peak_rss_mb"] = server.peak_rss_mb
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(report, indent=2))
    md = render(report)
    (args.out_dir / f"{args.name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
