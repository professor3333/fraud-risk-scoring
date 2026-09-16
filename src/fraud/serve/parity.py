"""Frozen-sample parity: the production artifact must reproduce stored probabilities.

``freeze`` writes, next to an artifact, a small sample of raw rows as request
payloads and the probabilities the artifact assigned to them at freeze time.
``verify`` rebuilds the rows through the production input path, re-scores them
with whatever object is loaded and fails loudly on any difference. The API runs
``verify`` at startup, so a service can only start on an artifact that gives
the same probability as the offline pipeline did for the same raw rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud.data import schema
from fraud.serve.frames import payloads_to_frame, row_to_payload

TOLERANCE = 1e-9


@dataclass(frozen=True)
class ParityResult:
    n_rows: int
    max_abs_diff: float
    artifact_sha256: str


def artifact_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_paths(artifact: Path) -> tuple[Path, Path]:
    stem = artifact.with_suffix("")
    return Path(f"{stem}_frozen_sample.json"), Path(f"{stem}_frozen_expected.json")


def manifest_path(artifact: Path) -> Path:
    """The promotion manifest next to a champion artifact (fraud.train.promotion)."""
    return artifact.with_name(f"{artifact.stem}_manifest.json")


def read_manifest(artifact: Path) -> dict[str, Any] | None:
    path = manifest_path(artifact)
    if not path.exists():
        return None
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def choose_sample(frame: pd.DataFrame, n_per_group: int = 25) -> pd.DataFrame:
    """A deterministic sample: the lowest TransactionIDs with and without identity."""
    ordered = frame.sort_values(schema.ID_COL)
    with_id = ordered[ordered[schema.HAS_IDENTITY_COL]].head(n_per_group)
    without = ordered[~ordered[schema.HAS_IDENTITY_COL]].head(n_per_group)
    return pd.concat([with_id, without]).reset_index(drop=True)


def freeze(model: Any, artifact: Path, sample: pd.DataFrame) -> tuple[Path, Path]:
    sample_path, expected_path = frozen_paths(artifact)
    payloads = [row_to_payload(row) for _, row in sample.iterrows()]
    sample_path.write_text(json.dumps(payloads))
    frame = payloads_to_frame(payloads)  # the production input path, not the training frame
    probabilities = np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
    expected = {
        "artifact_sha256": artifact_digest(artifact),
        "n_rows": int(len(sample)),
        "probabilities": {
            str(int(i)): float(p)
            for i, p in zip(sample[schema.ID_COL].to_numpy(), probabilities, strict=True)
        },
    }
    expected_path.write_text(json.dumps(expected, indent=2))
    return sample_path, expected_path


def verify(model: Any, artifact: Path, tolerance: float = TOLERANCE) -> ParityResult:
    """Raise ``RuntimeError`` unless ``model`` reproduces the frozen probabilities."""
    sample_path, expected_path = frozen_paths(artifact)
    if not sample_path.exists() or not expected_path.exists():
        raise FileNotFoundError(
            f"no frozen sample next to {artifact}; run scripts/freeze_artifact.py"
        )
    expected = json.loads(expected_path.read_text())
    digest = artifact_digest(artifact)
    if expected["artifact_sha256"] != digest:
        raise RuntimeError(
            f"artifact {artifact.name} ({digest[:12]}) is not the one the frozen sample was "
            f"made for ({expected['artifact_sha256'][:12]}); re-freeze or restore the artifact"
        )
    sample = payloads_to_frame(json.loads(sample_path.read_text()))
    got = np.asarray(model.predict_proba(sample)[:, 1], dtype=float)
    want = np.array(
        [expected["probabilities"][str(int(i))] for i in sample[schema.ID_COL].to_numpy()]
    )
    max_diff = float(np.max(np.abs(got - want))) if len(got) else 0.0
    if max_diff > tolerance:
        worst = int(np.argmax(np.abs(got - want)))
        raise RuntimeError(
            f"parity failure on {artifact.name}: max |diff| {max_diff:.3e} > {tolerance:.0e} "
            f"(TransactionID {int(sample[schema.ID_COL].iloc[worst])}: "
            f"expected {want[worst]:.9f}, got {got[worst]:.9f})"
        )
    return ParityResult(n_rows=len(sample), max_abs_diff=max_diff, artifact_sha256=digest)


def load_frozen_sample(artifact: Path) -> pd.DataFrame:
    return payloads_to_frame(json.loads(frozen_paths(artifact)[0].read_text()))
