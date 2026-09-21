"""Cut a release: checks → champion verified → git tag (→ deploy.yml deploys it).

The pushed tag runs .github/workflows/deploy.yml: Render rebuilds this tag's exact
commit (the deploy hook is pinned with ?ref=) without the weights, the service fetches
the champion from its GitHub release at startup, and the workflow fails unless the
live /health reports the commit it asked for; the Fly leg, when configured,
deploys the image this script can push with --fly-image. The weights never enter
the repository; they are built either here or by the retraining workflow
(.github/workflows/retrain.yml, ADR 0012) and reach the host as a digest-pinned
release asset.

    uv run python scripts/release.py v0.6.0              # checks, tag, push
    uv run python scripts/release.py v0.6.0 --dry-run    # what would happen
    uv run python scripts/release.py v0.6.0 --fly-image  # also build + push the Fly image
    uv run python scripts/release.py v0.6.0 --full-checks  # also require real-data tests

Preconditions: on main, clean tree, tag unused, pyproject version == tag without the
"v", models/champion present and reproducing its golden; for --fly-image, flyctl
logged in. The champion must already be published as a GitHub release
(scripts/publish_champion.py) for the hosted service to start.
Production releases require --full-checks on the machine holding the real dataset
and the production artifacts. Missing data, skipped tests or failing slow tests
block the tag. --skip-checks is not a substitute for this production gate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

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


def champion_published(tag: str) -> bool:
    """Whether the champion's own release exists, which is what the host fetches."""
    return (
        subprocess.run(["gh", "release", "view", tag], capture_output=True, cwd=ROOT).returncode
        == 0
    )


def check_slow_tests(*, dry_run: bool) -> None:
    """Require the real-data suite to pass without silently skipping missing inputs."""
    cmd = ("uv", "run", "pytest", "-q", "-m", "slow")
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return
    with TemporaryDirectory(prefix="fraud-release-") as tmp:
        report = Path(tmp) / "slow-tests.xml"
        sh(*cmd, "--junitxml", str(report))
        results = ET.parse(report)
        cases = list(results.iter("testcase"))
        if not cases or any(
            list(results.iter(outcome)) for outcome in ("skipped", "failure", "error")
        ):
            sys.exit(
                "full checks require passing slow tests with no skips; check data and artifacts"
            )
        print(f"full checks: {len(cases)} slow tests passed, none skipped", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="vX.Y.Z")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fly-image", action="store_true", help="also build and push the Fly image for the tag"
    )
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--skip-checks", action="store_true", help="CI already ran on this commit")
    checks.add_argument(
        "--full-checks",
        action="store_true",
        help="also require real-data tests to pass without skips",
    )
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
    champion_sha = parity.artifact_sha256[:12]
    print(f"champion {manifest['run_name']} sha {champion_sha}: parity ok")

    # The service fetches the champion from its own release, so a champion that was
    # promoted but never published would leave the host fetching the previous one.
    # The docstring has always required this; check it instead of assuming it.
    champion_tag = f"champion-{champion_sha}"
    if not champion_published(champion_tag):
        sys.exit(
            f"{champion_tag} is not published; run scripts/publish_champion.py first, then "
            "point the host's FRAUD_CHAMPION_URL at it (docs/promotion.md → Shipping a "
            "promoted champion). Without both, the service keeps serving the old champion."
        )
    print(f"{champion_tag}: published")

    # the ship sequence (§14): tests → lint → format → types
    if not args.skip_checks:
        print("$ uv run pytest -q", flush=True)
        if not args.dry_run:
            sh("uv", "run", "pytest", "-q")
        if args.full_checks:
            check_slow_tests(dry_run=args.dry_run)
        for cmd in (
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
