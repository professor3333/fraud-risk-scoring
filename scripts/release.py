"""Cut a release: checks → champion verified → git tag (→ deploy.yml deploys it).

The pushed tag runs .github/workflows/deploy.yml: Render rebuilds main (== the tag)
without the weights and the service fetches the champion from the private model repo
at startup; the Fly leg, when configured, deploys the image this script can push with
--fly-image. The weights never enter the repository or a public runner.

    uv run python scripts/release.py v0.6.0              # checks, tag, push
    uv run python scripts/release.py v0.6.0 --dry-run    # what would happen
    uv run python scripts/release.py v0.6.0 --fly-image  # also build + push the Fly image

Preconditions: on main, clean tree, tag unused, pyproject version == tag without the
"v", models/champion present and reproducing its golden; for --fly-image, flyctl
logged in. The champion must already be in the private model repo
(scripts/publish_champion.py) for the hosted service to start.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import joblib

from fraud.serve.parity import read_manifest, verify

ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "models" / "champion" / "model.joblib"


def sh(*cmd: str, check: bool = True) -> str:
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if check and out.returncode:
        sys.exit(f"$ {' '.join(cmd)}\n{out.stdout}{out.stderr}")
    return out.stdout.strip()


def fly_app() -> str:
    text = (ROOT / "fly.toml").read_text()
    m = re.search(r'^app\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        sys.exit("fly.toml has no app name")
    return m.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="vX.Y.Z")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fly-image", action="store_true", help="also build and push the Fly image for the tag"
    )
    parser.add_argument("--skip-checks", action="store_true", help="CI already ran on this commit")
    args = parser.parse_args()
    tag = args.tag
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        sys.exit("tag must look like v0.6.0")

    # the tree, the version, the tag (a dry run only reports these two)
    problems = []
    if sh("git", "branch", "--show-current") != "main":
        problems.append("release from main")
    if sh("git", "status", "--porcelain"):
        problems.append("working tree is not clean")
    for problem in problems:
        print(f"{'would stop: ' if args.dry_run else ''}{problem}")
    if problems and not args.dry_run:
        sys.exit(1)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if version != tag[1:]:
        sys.exit(f"pyproject.toml says {version}; set it to {tag[1:]} first (and uv lock)")
    if sh("git", "tag", "--list", tag):
        sys.exit(f"{tag} exists")
    sh("git", "fetch", "--tags", "origin")
    if sh("git", "ls-remote", "--tags", "origin", tag):
        sys.exit(f"{tag} exists on origin")

    # the champion
    manifest = read_manifest(CHAMPION)
    if manifest is None:
        sys.exit("models/champion has no manifest; run scripts/promote.py")
    parity = verify(joblib.load(CHAMPION), CHAMPION)
    print(f"champion {manifest['run_name']} sha {parity.artifact_sha256[:12]}: parity ok")

    # the ship sequence (§14): tests → lint → format → types
    if not args.skip_checks:
        for cmd in (
            ("uv", "run", "pytest", "-q"),
            ("uv", "run", "ruff", "check", "."),
            ("uv", "run", "ruff", "format", "--check", "."),
            ("uv", "run", "mypy"),
        ):
            print("$", " ".join(cmd), flush=True)
            if not args.dry_run:
                sh(*cmd)

    # optionally the Fly image, into Fly's private registry, labelled with the tag
    if args.fly_image:
        app = fly_app()
        cmd = ("flyctl", "deploy", "--app", app, "--build-only", "--push", "--image-label", tag)
        print("$", " ".join(cmd), f"→ registry.fly.io/{app}:{tag}", flush=True)
        if not args.dry_run:
            sh(*cmd)

    # the tag: pushing it runs deploy.yml
    message = f"{tag}: {manifest['model_version']} ({parity.artifact_sha256[:12]})"
    print(f"$ git tag -a {tag} -m {message!r} && git push origin {tag}")
    if not args.dry_run:
        sh("git", "tag", "-a", tag, "-m", message)
        sh("git", "push", "origin", tag)
        print(f"pushed {tag}; deploy.yml deploys it and runs deploy_check.py")


if __name__ == "__main__":
    main()
