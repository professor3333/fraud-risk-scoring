"""Freeze the monitoring reference for a served artifact.

Feature distributions from the training window; score distribution, action
shares and performance from the validation window. Written next to the artifact
as <run>_calibrated_monitor_reference.json (git-ignored with the models).

Example:
    uv run python scripts/monitor_reference.py --run-name xgb_f5_capacity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import yaml

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.monitor.reference import build_reference, save_reference
from fraud.serve.parity import artifact_digest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--serving-config", type=Path, default=ROOT / "configs" / "serving.yaml")
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    serving = yaml.safe_load(args.serving_config.read_text())
    artifact = args.models_dir / f"{args.run_name}_calibrated.joblib"
    model = joblib.load(artifact)
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    ref = build_reference(
        model,
        parts["train"],
        parts["validation"],
        float(serving["bands"]["block"]),
        int(serving.get("default_review_budget", 200)),
        artifact_digest(artifact),
    )
    path = save_reference(ref, artifact)
    perf = ref["performance"]
    print(f"wrote {path.name}")
    print(f"  reference actions {ref['actions']}")
    print(f"  reference PR-AUC {perf['pr_auc']:.4f}, block precision {perf['block_precision']:.3f}")


if __name__ == "__main__":
    main()
