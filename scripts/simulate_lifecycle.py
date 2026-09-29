"""Run the retraining lifecycle end to end on a simulated clock (ADR 0015).

The dataset ends at day 182 and has no future, so the lifecycle cannot run live: it
runs here, on a declared snapshot (the labelled Kaggle training data) and a declared
clock (the calendar days below), inside a scratch workspace that shares nothing with
the served champion, the real registry or the repository's configs. Every step uses
the same entry points the workflow does:

    day  92  scripts/retrain_cycle.py   bootstrap: no champion yet -> fit, gate, promote
             scripts/publish_champion.py   -> <workspace>/releases/champion-<sha12>/
             the policy change             -> <workspace>/policy/<day>.diff (serving.yaml)
             the service                   -> starts from that release + that config
    day 110  scripts/retrain_cycle.py   WAIT: 18 new training days, the step is 30
    day 122  scripts/retrain_cycle.py   calendar: fit through day 92, score challenger AND
                                        champion on days 93-122, gate, promote or keep
    day 183  scripts/retrain_cycle.py   WAIT: the month would read the reporting window

Labels are treated as final on the day they occur (`--label-maturity-days 0`): the
Kaggle labels already are, and under the host's 120-day rule no month of this dataset
is newer than the champion. That is the simulation's one assumption, and it is the
reason this is a simulation.

Example:
    uv run python scripts/simulate_lifecycle.py                 # ~25 min, real data
    uv run python scripts/simulate_lifecycle.py --model-config tests/…/light.yaml \
        --raw-dir tests/fixtures/raw --workspace /tmp/sim         # seconds, wiring only
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DAYS = (92, 110, 122, 183)


@dataclass
class Step:
    day: int
    trigger: str
    ran: bool
    promoted: bool
    reason: str
    champion_sha: str | None
    release: str | None
    policy_diff: str | None
    served: str | None
    seconds: float


def _cycle(ws: Path, day: int, args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    report_dir = ws / "steps" / f"day{day}"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "retrain_cycle.py"),
        "--as-of-day", str(day), "--label-maturity-days", "0",
        "--serving-config", str(ws / "configs" / "serving.yaml"),
        "--champion-dir", str(ws / "models" / "champion"),
        "--out-dir", str(ws / "candidates"),
        "--report-dir", str(report_dir),
        "--tracking-uri", f"sqlite:///{ws / 'mlflow.db'}",
        "--raw-dir", str(args.raw_dir),
        "--audit-db", str(ws / "no-audit.sqlite"),
    ]  # fmt: skip
    if args.cache_dir is not None:
        cmd += ["--cache-dir", str(args.cache_dir)]
    if args.model_config is not None:
        cmd += ["--model-config", str(args.model_config)]
    if args.promotion is not None:
        cmd += ["--promotion", str(args.promotion)]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "GITHUB_OUTPUT": ""})
    seconds = time.perf_counter() - t0
    ran = sorted(report_dir.glob("cycle_day*.json"))
    path = ran[-1] if ran else report_dir / "last_decision.json"
    return json.loads(path.read_text()), seconds


def _publish(ws: Path) -> str:
    out = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "publish_champion.py"),
            "--champion-dir", str(ws / "models" / "champion"),
            "--to-dir", str(ws / "releases"),
        ],
        check=True, cwd=ROOT, capture_output=True, text=True,
    ).stdout  # fmt: skip
    return str(json.loads(out.strip().splitlines()[-1])["FRAUD_CHAMPION_URL"])


def _serve(ws: Path, release_url: str, day: int) -> str:
    """Start the service the way the host does: an empty image, the release URL, and the
    reviewed config. It fetches, pins, pairs bands with the digest and checks parity."""
    from fraud.serve.app import load_state

    deploy = ws / "deploy" / f"day{day}"
    (deploy / "configs").mkdir(parents=True)
    text = (ws / "configs" / "serving.yaml").read_text()
    lines = [ln for ln in text.splitlines() if not ln.startswith(("audit_db:", "model_path:"))]
    lines += ["model_path: models/champion/model.joblib", "audit_db: null"]
    (deploy / "configs" / "serving.yaml").write_text("\n".join(lines) + "\n")
    previous = os.environ.get("FRAUD_CHAMPION_URL")
    os.environ["FRAUD_CHAMPION_URL"] = release_url
    try:
        state = load_state(deploy / "configs" / "serving.yaml")
    finally:
        if previous is None:
            os.environ.pop("FRAUD_CHAMPION_URL", None)
        else:
            os.environ["FRAUD_CHAMPION_URL"] = previous
    return f"{state.model_version} · bands {state.bands.review:.4f}/{state.bands.block:.4f}"


def run(args: argparse.Namespace, ws: Path) -> list[Step]:
    (ws / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "configs" / "serving.yaml", ws / "configs" / "serving.yaml")
    (ws / "policy").mkdir(exist_ok=True)
    # MLflow puts a new experiment's artifacts under ./mlruns of whatever directory the
    # run starts in, i.e. the repository's own store. Created here first, they stay here.
    from mlflow.tracking import MlflowClient

    MlflowClient(f"sqlite:///{ws / 'mlflow.db'}").create_experiment(
        "fraud-retrain-cycle", artifact_location=(ws / "mlruns").as_uri()
    )
    steps: list[Step] = []
    for day in args.days:
        before = (ws / "configs" / "serving.yaml").read_text()
        record, seconds = _cycle(ws, day, args)
        trigger = record["trigger"]
        cycle = record.get("cycle")
        promoted = bool(record.get("promoted"))
        release = diff = served = sha = None
        if promoted:
            sha = str(record["manifest"]["artifact_sha256"])[:12]
            release = _publish(ws)
            after = (ws / "configs" / "serving.yaml").read_text()
            diff_path = ws / "policy" / f"day{day}.diff"
            diff_path.write_text(
                "".join(
                    difflib.unified_diff(
                        before.splitlines(keepends=True), after.splitlines(keepends=True),
                        "configs/serving.yaml", "configs/serving.yaml",
                    )
                )
            )  # fmt: skip
            diff = str(diff_path)
            served = _serve(ws, release, day)
        steps.append(
            Step(
                day=day,
                trigger=str(trigger["rule"]),
                ran=cycle is not None,
                promoted=promoted,
                reason=str(cycle["reason"]) if cycle else str(trigger["reason"]),
                champion_sha=sha,
                release=release,
                policy_diff=diff,
                served=served,
                seconds=seconds,
            )
        )
        print(f"day {day}: {steps[-1].trigger} · {'PROMOTED' if promoted else 'no promotion'}")
    return steps


def render(steps: list[Step], args: argparse.Namespace, ws: Path) -> str:
    lines = [
        "# Simulated retraining lifecycle",
        "",
        f"Snapshot `{args.raw_dir}`; clock days {', '.join(str(d) for d in args.days)}; "
        "labels final on the day they occur (`--label-maturity-days 0`); workspace "
        f"`{ws}`. Nothing outside the workspace was read or written except the data cache.",
        "",
        "| day | trigger | ran | outcome | published | service started from it | seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in steps:
        outcome = "PROMOTED" if s.promoted else ("kept champion" if s.ran else "waited")
        lines.append(
            f"| {s.day} | `{s.trigger}` | {'yes' if s.ran else 'no'} | {outcome} | "
            f"{'champion-' + s.champion_sha if s.champion_sha else '—'} | {s.served or '—'} | "
            f"{s.seconds:.0f} |"
        )
    lines += ["", "## Why each step did what it did", ""]
    lines += [f"- **day {s.day}** — {s.reason}" for s in steps]
    diffs = [s for s in steps if s.policy_diff]
    for s in diffs:
        lines += ["", f"## Policy change proposed on day {s.day}", "", "```diff"]
        lines += [Path(s.policy_diff).read_text().rstrip(), "```"]  # type: ignore[arg-type]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=lambda v: tuple(int(d) for d in v.split(",")), default=DAYS)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--model-config", type=Path, default=None, help="challenger recipe")
    parser.add_argument("--promotion", type=Path, default=None, help="promotion.yaml override")
    parser.add_argument("--workspace", type=Path, default=None, help="default: a new temp dir")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports" / "retrain" / "simulation.md"
    )
    args = parser.parse_args()
    ws = args.workspace or Path(tempfile.mkdtemp(prefix="lifecycle-"))
    if ws.exists() and any(ws.iterdir()):
        raise SystemExit(f"{ws} is not empty: the simulation starts from nothing")
    ws.mkdir(parents=True, exist_ok=True)
    steps = run(args, ws)
    text = render(steps, args, ws)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
