"""Upload models/champion/ to a private Hugging Face model repo for the hosted service.

A host that builds from the public repository must not get the weights; the service
fetches them at startup from this private repo with a token (FRAUD_CHAMPION_URL +
HF_TOKEN; docs/deployment.md). Uses the
huggingface_hub CLI through uvx, so nothing is added to the project's dependencies.

    uv run python scripts/publish_champion.py --repo <user>/fraud-risk-scoring-model
    uv run python scripts/publish_champion.py --repo … --dry-run

Needs `uvx --from huggingface_hub hf auth login` once (or HF_TOKEN in the environment).
Prints the FRAUD_CHAMPION_URL to set on the host.
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
    parser.add_argument("--repo", required=True, help="<user>/<name> of the private model repo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = read_manifest(CHAMPION_DIR / "model.joblib")
    if manifest is None:
        sys.exit("models/champion has no manifest; run scripts/promote.py first")
    parity = verify(joblib.load(CHAMPION_DIR / "model.joblib"), CHAMPION_DIR / "model.joblib")
    missing = [f for f in FILES if not (CHAMPION_DIR / f).exists()]
    if missing:
        sys.exit(f"champion directory is missing {missing}")
    print(f"champion {manifest['run_name']} sha {parity.artifact_sha256[:12]}: parity ok")

    hf = ["uvx", "--from", "huggingface_hub", "hf"]
    commands = [
        [*hf, "repo", "create", args.repo, "--repo-type", "model", "--private", "--exist-ok"],
        [*hf, "upload", args.repo, str(CHAMPION_DIR), ".", "--repo-type", "model",
         "--commit-message", f"{manifest['model_version']} ({parity.artifact_sha256[:12]})"],
    ]  # fmt: skip
    for cmd in commands:
        print("$", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)
    url = f"https://huggingface.co/{args.repo}/resolve/main"
    print(json.dumps({"FRAUD_CHAMPION_URL": url, "files": list(FILES)}, indent=2))
    print("set FRAUD_CHAMPION_URL and HF_TOKEN (a read token) on the hosted service")


if __name__ == "__main__":
    main()
