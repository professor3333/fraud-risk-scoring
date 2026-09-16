"""Promote a candidate to champion if it clears every acceptance gate (docs/promotion.md).

    candidate (models/<run>_calibrated.joblib + frozen golden)
      -> metrics on the validation window under its own re-derived policy
      -> gates against the current champion (configs/promotion.yaml) + parity
      -> registry version (MLflow model registry, tags carry the verdict)
      -> if every gate passes: models/champion/ is replaced and the alias moves

Example:
    uv run python scripts/promote.py --run-name xgb_f5_capacity
    uv run python scripts/promote.py --run-name xgb_f5_interactions --dry-run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import yaml

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.policy import load_policy_config
from fraud.evaluate.threshold import load_threshold_config
from fraud.serve.parity import read_manifest, verify
from fraud.train.promotion import (
    CHAMPION_ARTIFACT,
    all_pass,
    build_manifest,
    candidate_metrics,
    load_promotion_config,
    materialise,
    register_candidate,
    run_gates,
)

ROOT = Path(__file__).resolve().parents[1]


def find_model_config(run_name: str) -> Path:
    """The training config whose run_name produced this artifact."""
    matches = [
        path
        for path in sorted((ROOT / "configs" / "model").glob("*.yaml"))
        if yaml.safe_load(path.read_text()).get("run_name") == run_name
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one configs/model/*.yaml with run_name {run_name}: {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "promotion.yaml")
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "split.yaml")
    parser.add_argument("--threshold", type=Path, default=ROOT / "configs" / "threshold.yaml")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs" / "policy.yaml")
    parser.add_argument(
        "--model-config", type=Path, default=None,
        help="training config; default: the configs/model/*.yaml with this run_name",
    )  # fmt: skip
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "promotion")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument(
        "--dry-run", action="store_true", help="evaluate and register; never promote"
    )
    args = parser.parse_args()

    cfg = load_promotion_config(args.config)
    champion_dir = ROOT / cfg.champion_dir
    cfg = replace(cfg, champion_dir=champion_dir)
    candidate = args.models_dir / f"{args.run_name}_calibrated.joblib"
    model = joblib.load(candidate)
    parity = verify(model, candidate)  # a candidate without a reproducible golden is not one
    print(
        f"candidate {candidate.name} sha {parity.artifact_sha256[:12]}: parity {parity.n_rows} rows"
    )

    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(args.split),
    )
    metrics = candidate_metrics(
        model, parts["validation"], load_threshold_config(args.threshold),
        load_policy_config(args.policy), cfg,
    )  # fmt: skip
    current = read_manifest(champion_dir / CHAMPION_ARTIFACT)
    champion_metrics = None if current is None else current["metrics"]
    gates = run_gates(metrics, champion_metrics, cfg.gates)
    passed = all_pass(gates)
    for g in gates:
        print(f"  {'PASS' if g.passed else 'FAIL'} {g.metric:<24} {g.candidate:.4f}  {g.reason}")
    if (
        current is not None
        and current["run_name"] == args.run_name
        and current.get("artifact_sha256") == parity.artifact_sha256
    ):
        print("this artifact is already the champion")
    promote = passed and not args.dry_run

    tc = yaml.safe_load((args.model_config or find_model_config(args.run_name)).read_text())
    windows = yaml.safe_load(args.split.read_text())["windows"]
    info = {
        "model": tc["model"]["type"],
        "experiment": tc.get("experiment", ""),
        "feature_set": Path(tc["features"]).stem,
        "primary_metric": "pr_auc",
        "calibration": model.method,
        "training_window_days": [windows["train"]["start_day"], windows["train"]["end_day"]],
        "validation_window_days": [
            windows["validation"]["start_day"], windows["validation"]["end_day"],
        ],
    }  # fmt: skip
    version = register_candidate(args.tracking_uri, cfg, candidate, args.run_name, gates, promote)
    manifest = build_manifest(args.run_name, candidate, metrics, gates, info, version)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "candidate": args.run_name,
        "champion_before": None if current is None else current["run_name"],
        "registry_version": version,
        "passed": passed,
        "promoted": promote,
        "dry_run": args.dry_run,
        "gates": [asdict(g) for g in gates],
        "metrics": metrics,
    }
    (args.report_dir / f"{args.run_name}.json").write_text(json.dumps(report, indent=2))
    if promote:
        target = materialise(candidate, model, cfg, manifest, parts["train"], parts["validation"])
        print(
            f"PROMOTED: {target} (registry version {version}, alias champion); bands"
            f" review {metrics['review_threshold']:.3f} / block {metrics['block_threshold']:.3f}"
        )
    elif passed:
        print(f"passed every gate; not promoted (dry run). registry version {version}")
    else:
        failed = [g.metric for g in gates if not g.passed]
        print(f"REJECTED on {', '.join(failed)}; registry version {version}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
