"""Exercise release orchestration without touching Git, models or the network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def release_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ModuleType, list[tuple[str, ...]]]:
    path = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("release_script", path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    monkeypatch.setattr(
        script, "read_manifest", lambda _: {"run_name": "fixture", "model_version": "fixture"}
    )
    monkeypatch.setattr(script.joblib, "load", lambda _: object())
    monkeypatch.setattr(script, "verify", lambda *_: SimpleNamespace(artifact_sha256="a" * 64))
    # the champion's release is a `gh` call; tests stay offline and assume it published
    monkeypatch.setattr(script, "champion_published", lambda _: True)
    calls: list[tuple[str, ...]] = []

    def sh(*cmd: str) -> str:
        calls.append(cmd)
        if cmd == ("git", "branch", "--show-current"):
            return "main"
        if "--junitxml" in cmd:
            Path(cmd[-1]).write_text("<testsuites><testsuite><testcase /></testsuite></testsuites>")
        return ""

    monkeypatch.setattr(script, "sh", sh)
    return script, calls


@pytest.mark.parametrize("mode", [[], ["--full-checks"], ["--skip-checks"]])
def test_release_checks_run_before_tagging(
    release_env: tuple[ModuleType, list[tuple[str, ...]]],
    monkeypatch: pytest.MonkeyPatch,
    mode: list[str],
) -> None:
    script, calls = release_env
    monkeypatch.setattr(sys, "argv", ["release.py", "v1.2.3", *mode])
    script.main()
    pytest_calls = [cmd for cmd in calls if cmd[:3] == ("uv", "run", "pytest")]
    assert len(pytest_calls) == (0 if "--skip-checks" in mode else 1 + ("--full-checks" in mode))
    if "--full-checks" in mode:
        assert pytest_calls[1][3:6] == ("-q", "-m", "slow")
        assert calls.index(pytest_calls[1]) < calls.index(("uv", "run", "ruff", "check", "."))
    assert calls[-2][:3] == ("git", "tag", "-a")
    assert calls[-1] == ("git", "push", "origin", "v1.2.3")


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error", "empty", "exit"])
def test_incomplete_full_checks_cannot_tag_or_push(
    release_env: tuple[ModuleType, list[tuple[str, ...]]],
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    script, calls = release_env
    original = script.sh

    def sh(*cmd: str) -> str:
        result = original(*cmd)
        if "--junitxml" in cmd:
            if outcome == "exit":
                raise SystemExit("slow pytest failed")
            case = "" if outcome == "empty" else f"<testcase><{outcome} /></testcase>"
            Path(cmd[-1]).write_text(f"<testsuites><testsuite>{case}</testsuite></testsuites>")
        return result

    monkeypatch.setattr(script, "sh", sh)
    monkeypatch.setattr(sys, "argv", ["release.py", "v1.2.3", "--full-checks"])
    with pytest.raises(SystemExit, match="slow|full checks"):
        script.main()
    assert not any(cmd[:3] == ("git", "tag", "-a") or cmd[:2] == ("git", "push") for cmd in calls)


def test_full_checks_cannot_be_combined_with_skip_checks(
    release_env: tuple[ModuleType, list[tuple[str, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    script, calls = release_env
    monkeypatch.setattr(sys, "argv", ["release.py", "v1.2.3", "--full-checks", "--skip-checks"])
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == 2
    assert calls == []


def test_full_checks_dry_run_only_prints_checks_and_tag(
    release_env: tuple[ModuleType, list[tuple[str, ...]]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script, calls = release_env
    monkeypatch.setattr(sys, "argv", ["release.py", "v1.2.3", "--full-checks", "--dry-run"])
    script.main()
    assert "$ uv run pytest -q -m slow" in capsys.readouterr().out
    assert not any(cmd[0] == "uv" for cmd in calls)
    assert not any(cmd[:3] == ("git", "tag", "-a") or cmd[:2] == ("git", "push") for cmd in calls)


def test_an_unpublished_champion_cannot_be_tagged(
    release_env: tuple[ModuleType, list[tuple[str, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Promotion writes models/champion/; the host fetches the champion from its own
    release. Tagging a champion that was never published would ship a release whose
    service keeps serving the previous one, silently (docs/promotion.md)."""
    script, calls = release_env
    monkeypatch.setattr(script, "champion_published", lambda _: False)
    monkeypatch.setattr(sys, "argv", ["release.py", "v1.2.3", "--skip-checks"])
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert "champion-aaaaaaaaaaaa is not published" in str(exc.value)
    assert not [c for c in calls if c[:3] == ("git", "tag", "-a")], "created the tag anyway"
    assert not [c for c in calls if c[:2] == ("git", "push")], "pushed anyway"
