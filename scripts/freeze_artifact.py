"""Freeze a production artifact: store a raw sample and the probabilities it produces.

Writes models/<run>_calibrated_frozen_sample.json and
models/<run>_calibrated_frozen_expected.json (git-ignored with the artifact). The
API verifies against them at startup; tests/test_parity.py checks them too.

Example:
    uv run python scripts/freeze_artifact.py --run-name xgb_f5_interactions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.serve.parity import choose_sample, freeze, verify

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--n-per-group", type=int, default=25)
    args = parser.parse_args()

    artifact = args.models_dir / f"{args.run_name}_calibrated.joblib"
    model = joblib.load(artifact)
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    sample = choose_sample(parts["validation"], args.n_per_group)
    sample_path, expected_path = freeze(model, artifact, sample)
    result = verify(joblib.load(artifact), artifact)
    print(f"froze {result.n_rows} rows -> {sample_path.name}, {expected_path.name}")
    sha = result.artifact_sha256[:12]
    print(f"artifact sha256 {sha}; re-verified, max |diff| {result.max_abs_diff:.1e}")


if __name__ == "__main__":
    main()
