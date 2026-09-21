"""The reviewed half of a promotion: the policy edit a new champion implies (ADR 0012).

Almost everything about a champion travels with it — the artifact, its golden, its
monitoring reference and a manifest carrying the model's version and facts. Two
things do not, and deliberately:

- **the policy bands.** Block and review decide what happens to a customer's
  transaction (ADR 0006). The service refuses to take them from a manifest it
  fetched over the network, because that manifest is not digest-pinned; they come
  from `configs/serving.yaml`, which ships in the image from a reviewed commit
  (`docs/security.md`).
- **which artifact those bands belong to** — `champion_sha256`, the digest the
  service must find when it loads the champion.

A retrained champion re-derives its block threshold on its own month, so it almost
always needs both edited. An unattended job therefore cannot finish a deployment on
its own: it writes this patch and opens a pull request, and a human merging that
diff is the approval. The rest of the pipeline is automatic; the policy change is
reviewed, which is the right place to put the one human step.
"""

from __future__ import annotations

import re
from typing import Any

BANDS_KEYS = ("review", "block")


def _replace_value(line: str, key: str, value: str) -> str:
    """Rewrite ``key: old`` keeping the indentation and any trailing comment."""
    match = re.match(rf"^(\s*{re.escape(key)}:\s*)(\S+)(.*)$", line)
    if match is None:
        raise ValueError(f"line does not set {key}: {line!r}")
    return f"{match.group(1)}{value}{match.group(3)}"


def read_setting(text: str, key: str) -> str | None:
    """A top-level scalar from the config text, or None (no YAML round-trip)."""
    for line in text.splitlines():
        match = re.match(rf"^{re.escape(key)}:\s*(\S+)", line)
        if match:
            return match.group(1).strip("'\"")
    return None


def update_serving_config(text: str, manifest: dict[str, Any]) -> str:
    """`configs/serving.yaml` with this champion's bands, version and digest.

    Line edits, not a YAML round-trip: the file is mostly comments explaining why
    each value is what it is, and a dump would delete all of them. Raises if the
    file does not have the shape those comments describe.
    """
    bands = manifest["bands"]
    digest = str(manifest["artifact_sha256"])[:12]
    lines = text.splitlines()
    out: list[str] = []
    in_bands = False
    seen: set[str] = set()
    for line in lines:
        if re.match(r"^bands:\s*$", line):
            in_bands = True
            out.append(line)
            continue
        if in_bands:
            stripped = line.strip()
            key = stripped.split(":", 1)[0] if ":" in stripped else ""
            if key in BANDS_KEYS and line.startswith((" ", "\t")):
                out.append(_replace_value(line, key, repr(float(bands[key]))))
                seen.add(key)
                continue
            if stripped and not line.startswith((" ", "\t")):
                in_bands = False  # the block ended
        if re.match(r"^model_version:\s*", line):
            out.append(_replace_value(line, "model_version", str(manifest["model_version"])))
            seen.add("model_version")
            continue
        if re.match(r"^champion_sha256:\s*", line):
            out.append(_replace_value(line, "champion_sha256", digest))
            seen.add("champion_sha256")
            continue
        out.append(line)

    missing = {"review", "block", "model_version"} - seen
    if missing:
        raise ValueError(f"configs/serving.yaml does not set {sorted(missing)}")
    if "champion_sha256" not in seen:
        out = _insert_after(
            out,
            "model_version:",
            [
                "# The artifact these bands were measured on: the service refuses to start on any",
                "# other (scripts/retrain_cycle.py writes this; ADR 0012). Remove it only to serve",
                "# an artifact nobody has reviewed the policy for.",
                f"champion_sha256: {digest}",
            ],
        )
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _insert_after(lines: list[str], prefix: str, block: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            return [*lines[: i + 1], *block, *lines[i + 1 :]]
    raise ValueError(f"no line starts with {prefix!r}")
