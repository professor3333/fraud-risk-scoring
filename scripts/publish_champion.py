"""Publish models/champion/ as the assets of a GitHub release for the hosted service.

A host that builds from the repository gets an image without weights; the service
fetches them at startup from FRAUD_CHAMPION_URL (docs/deployment.md). The store is a
release on this repository tagged ``champion-<sha12>`` — one immutable release per
promoted champion, marked pre-release so it never becomes the repository's "Latest" —
whose assets are the five champion files. Public, free, no token: the artifact is a
portfolio model on public data, and the ``<sha12>`` in the tag is what the service
pins the download against before it deserializes it (``fraud.serve.app``). The parity
check cannot serve that purpose: it runs after ``joblib.load`` has already executed
the pickle, and compares against a golden fetched from the same store.

    uv run python scripts/publish_champion.py
    uv run python scripts/publish_champion.py --dry-run

Needs ``gh auth login`` once. Prints the FRAUD_CHAMPION_URL to set on the host.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib

from fraud.serve.parity import read_manifest, verify

ROOT = Path(__file__).resolve().parents[1]
CHAMPION_DIR = ROOT / "models" / "champion"
FILES = (
    "model.joblib",
    "model_frozen_sample.json",
    "model_frozen_expected.json",
    "model_manifest.json",
    "model_monitor_reference.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = read_manifest(CHAMPION_DIR / "model.joblib")
    if manifest is None:
        sys.exit("models/champion has no manifest; run scripts/promote.py first")
    parity = verify(joblib.load(CHAMPION_DIR / "model.joblib"), CHAMPION_DIR / "model.joblib")
    missing = [f for f in FILES if not (CHAMPION_DIR / f).exists()]
    if missing:
        sys.exit(f"champion directory is missing {missing}")
    sha = parity.artifact_sha256[:12]
    print(f"champion {manifest['run_name']} sha {sha}: parity ok")

    repo = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        check=True, capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip()  # fmt: skip
    tag = f"champion-{sha}"
    url = f"https://github.com/{repo}/releases/download/{tag}"
    exists = subprocess.run(
        ["gh", "release", "view", tag], capture_output=True, cwd=ROOT
    ).returncode == 0  # fmt: skip
    if exists:
        print(f"release {tag} already exists; assets are immutable, nothing to do")
    else:
        cmd = [
            "gh", "release", "create", tag, *[str(CHAMPION_DIR / f) for f in FILES],
            "--prerelease", "--title", f"champion {manifest['model_version']}",
            "--notes", f"Served weights: {manifest['run_name']} ({manifest['model_version']}). "
            "Assets of this release are what the hosted service fetches at startup "
            "(FRAUD_CHAMPION_URL); the tag digest is verified before the artifact is loaded.",
        ]  # fmt: skip
        print("$", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)
    print(json.dumps({"FRAUD_CHAMPION_URL": url, "files": list(FILES)}, indent=2))
    print("set FRAUD_CHAMPION_URL on the hosted service (no token needed)")


if __name__ == "__main__":
    main()
