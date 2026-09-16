"""Subgroup robustness of the champion on the validation window (docs/subgroups.md).

For each grouping (product, card network / type, identity, e-mail family, device,
amount band, ten-day block): n, fraud prevalence, PR-AUC, block precision, recall
at block and at block + review, flag rate — under the served policy's thresholds,
held fixed. Ranking metrics are suppressed below --min-positives.

Example:
    uv run python scripts/subgroups.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.subgroups import findings, subgroup_table
from fraud.serve.parity import read_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-config", type=Path, default=ROOT / "configs" / "serving.yaml")
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--min-positives", type=int, default=30)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "subgroups")
    args = parser.parse_args()

    serving = yaml.safe_load(args.serving_config.read_text())
    artifact = ROOT / serving["model_path"]
    manifest = read_manifest(artifact)
    bands = serving["bands"] if manifest is None else manifest["bands"]
    model = joblib.load(artifact)
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    val = parts["validation"]
    p = np.asarray(model.predict_proba(val)[:, 1], dtype=float)
    table = subgroup_table(
        val, p, float(bands["block"]), float(bands["review"]), args.min_positives, args.min_rows
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "validation.csv", index=False)
    notes = findings(table)
    (args.out_dir / "findings.md").write_text(
        "# Subgroup findings (validation, served policy)\n\n"
        + ("\n".join(f"- {n}" for n in notes) if notes else "- none flagged")
        + "\n"
    )
    pd.set_option("display.width", 250)
    cols = [
        "grouping", "level", "n", "share_of_fraud", "prevalence", "mean_score", "pr_auc",
        "delta_pr_auc", "block_precision", "recall_block", "recall_block_plus_review", "flag_rate",
    ]  # fmt: skip
    print(table[cols].round(3).to_string(index=False))
    print("\nfindings:")
    print("\n".join(f"- {n}" for n in notes) if notes else "- none flagged")


if __name__ == "__main__":
    main()
