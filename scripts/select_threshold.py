"""Choose the operating threshold from the cost model (ADR 0006).

Primary curve: the calibrated model's probabilities on the validation window.
Cross-check: out-of-fold training-window scores written by scripts/calibrate.py,
passed through the same calibrator, so a threshold exists that never saw
validation.

Example:
    uv run python scripts/select_threshold.py --run-name xgb_v2_tuned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.threshold import (
    cost_at,
    cost_curve,
    load_threshold_config,
    plot_cost_curve,
    select_threshold,
    sensitivity_table,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="e.g. xgb_v2_tuned")
    parser.add_argument(
        "--threshold-config", type=Path, default=ROOT / "configs" / "threshold.yaml"
    )
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "threshold")
    args = parser.parse_args()

    tcfg = load_threshold_config(args.threshold_config)
    model = joblib.load(args.models_dir / f"{args.run_name}_calibrated.joblib")
    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    val = parts["validation"]
    y, amt = val["isFraud"].to_numpy(), val["TransactionAmt"].to_numpy()
    p = model.predict_proba(val)[:, 1]

    curve = cost_curve(y, p, amt, tcfg.costs, tcfg.thresholds)
    chosen = select_threshold(curve)
    f1_best = float(curve.loc[curve["f1"].idxmax(), "threshold"])
    sens = sensitivity_table(y, p, amt, tcfg.costs, tcfg.thresholds)

    oof = pd.read_csv(ROOT / "reports" / "calibration" / f"{args.run_name}_oof.csv")
    oof_p = model.calibrate_scores(oof["score"].to_numpy())
    oof_curve = cost_curve(oof["y"], oof_p, oof["TransactionAmt"], tcfg.costs, tcfg.thresholds)
    oof_chosen = select_threshold(oof_curve)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.out_dir / f"{args.run_name}_cost_curve_validation.csv", index=False)
    oof_curve.to_csv(args.out_dir / f"{args.run_name}_cost_curve_oof.csv", index=False)
    sens.to_csv(args.out_dir / f"{args.run_name}_sensitivity.csv", index=False)
    plot_cost_curve(curve, chosen, f"{args.run_name} / validation").savefig(
        args.out_dir / f"{args.run_name}_cost_curve.png", dpi=120
    )
    summary = {
        "chosen_threshold_validation": chosen,
        "chosen_threshold_oof_crosscheck": oof_chosen,
        "at_chosen": cost_at(curve, chosen),
        "at_oof_choice_on_validation": cost_at(curve, oof_chosen),
        "at_0.5": cost_at(curve, 0.5),
        "at_f1_optimum": cost_at(curve, f1_best),
        "cost_no_model_approve_all": float(curve["fn_cost"].max()),
    }
    (args.out_dir / f"{args.run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
