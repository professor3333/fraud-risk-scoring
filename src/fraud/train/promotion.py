"""Candidate → acceptance gates → champion (ADR 0010, docs/promotion.md).

The service loads whatever sits in the champion directory; nothing else names an
artifact. A candidate becomes the champion only by clearing every gate against
the current champion on the validation window and reproducing its frozen
golden. The MLflow model registry records every candidate, its gate results and
which version is the champion; the champion directory is that version,
materialised: the artifact, its golden, its monitoring reference and a manifest
the service reads for its version, facts and policy bands.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fraud.data import schema
from fraud.evaluate.calibration import calibration_metrics
from fraud.evaluate.metrics import compute_metrics, top_k_per_day
from fraud.evaluate.policy import (
    Policy,
    PolicyConfig,
    block_threshold,
    evaluate_policy,
    operating_points,
    size_review_band,
)
from fraud.evaluate.threshold import ThresholdConfig
from fraud.monitor.reference import build_reference, save_reference
from fraud.serve.parity import (
    artifact_digest,
    frozen_paths,
    manifest_path,
    verify,
)

SECONDS_PER_DAY = 86_400
CHAMPION_ARTIFACT = "model.joblib"


@dataclass(frozen=True)
class Gate:
    metric: str
    higher_is_better: bool
    tolerance_vs_champion: float | None = None  # how far below (above) the champion is allowed
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class PromotionConfig:
    registry_model: str
    champion_dir: Path
    review_budget_per_day: int
    block_min_precision: float
    gates: tuple[Gate, ...]


def load_promotion_config(path: Path) -> PromotionConfig:
    raw = yaml.safe_load(path.read_text())
    gates = tuple(
        Gate(
            metric=name,
            higher_is_better=bool(g["higher_is_better"]),
            tolerance_vs_champion=(
                None
                if g.get("tolerance_vs_champion") is None
                else float(g["tolerance_vs_champion"])
            ),
            min=None if g.get("min") is None else float(g["min"]),
            max=None if g.get("max") is None else float(g["max"]),
        )
        for name, g in raw["gates"].items()
    )
    return PromotionConfig(
        registry_model=str(raw["registry_model"]),
        champion_dir=Path(raw["champion_dir"]),
        review_budget_per_day=int(raw["review_budget_per_day"]),
        block_min_precision=float(raw["block_min_precision"]),
        gates=gates,
    )


# --- what a candidate is measured on ----------------------------------------------


def candidate_metrics(
    model: Any,
    validation: pd.DataFrame,
    tcfg: ThresholdConfig,
    pcfg: PolicyConfig,
    cfg: PromotionConfig,
) -> dict[str, float]:
    """Every number a gate may refer to, from the validation window only.

    The block threshold is re-selected at the precision bar and the review band
    sized to the budget, so the cost and recall figures are those of the policy
    this candidate would actually serve under, not the champion's thresholds.
    """
    y = validation[schema.TARGET_COL].to_numpy(dtype=int)
    p = np.asarray(model.predict_proba(validation)[:, 1], dtype=float)
    amount = validation["TransactionAmt"].to_numpy(dtype=float)
    day = (validation[schema.TIME_COL] // SECONDS_PER_DAY).to_numpy()
    n_days = len(np.unique(day))
    k = cfg.review_budget_per_day
    points = operating_points(y, p, n_days, tcfg.thresholds)
    block_t = block_threshold(points, cfg.block_min_precision)
    review_t = size_review_band(p, day, block_t, k)
    policy = evaluate_policy(
        y, p, amount, day, Policy(block_t, review_t, k), tcfg.costs, pcfg.review
    )
    ranking = compute_metrics(y, p, block_t)
    cal = calibration_metrics(y, p)
    top = top_k_per_day(y, p, day, k)
    return {
        "pr_auc": ranking["pr_auc"],
        "roc_auc": ranking["roc_auc"],
        "brier": cal["brier"],
        "ece": cal["ece"],
        f"recall_at_{k}_per_day": top[f"recall_at_{k}_per_day"],
        f"precision_at_{k}_per_day": top[f"precision_at_{k}_per_day"],
        "block_threshold": block_t,
        "review_threshold": review_t,
        "block_precision": policy["block_precision"],
        "recall_block": policy["recall_block"],
        "recall_block_plus_review": policy["recall_block_plus_review"],
        "cost_per_transaction": policy["total_cost"] / len(y),
        "n_validation": float(len(y)),
        "positives": float(y.sum()),
    }


# --- gates ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    metric: str
    candidate: float
    champion: float | None
    passed: bool
    reason: str


def run_gates(
    candidate: dict[str, float], champion: dict[str, float] | None, gates: tuple[Gate, ...]
) -> list[GateResult]:
    """Absolute bounds always apply; the champion comparison applies when there is one."""
    results: list[GateResult] = []
    for g in gates:
        value = candidate[g.metric]
        ref = None if champion is None else champion.get(g.metric)
        reasons: list[str] = []
        ok = True
        if g.min is not None and value < g.min:
            ok, reasons = False, [*reasons, f"{value:.4f} < min {g.min}"]
        if g.max is not None and value > g.max:
            ok, reasons = False, [*reasons, f"{value:.4f} > max {g.max}"]
        if g.tolerance_vs_champion is not None and ref is not None:
            if g.higher_is_better and value < ref - g.tolerance_vs_champion:
                ok = False
                reasons.append(f"{value:.4f} < champion {ref:.4f} - {g.tolerance_vs_champion}")
            if not g.higher_is_better and value > ref + g.tolerance_vs_champion:
                ok = False
                reasons.append(f"{value:.4f} > champion {ref:.4f} + {g.tolerance_vs_champion}")
        if ok:
            reasons = ["ok" if ref is None else f"vs champion {ref:.4f}"]
        results.append(GateResult(g.metric, float(value), ref, ok, "; ".join(reasons)))
    return results


def all_pass(results: list[GateResult]) -> bool:
    return all(r.passed for r in results)


# --- the champion directory ----------------------------------------------------------


def materialise(
    candidate_artifact: Path,
    model: Any,
    cfg: PromotionConfig,
    manifest: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> Path:
    """Copy the artifact and its golden into the champion directory; freeze the
    monitoring reference there; write the manifest. Verified before it is returned."""
    cfg.champion_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.champion_dir / CHAMPION_ARTIFACT
    shutil.copyfile(candidate_artifact, target)
    for src, dst in zip(frozen_paths(candidate_artifact), frozen_paths(target), strict=True):
        shutil.copyfile(src, dst)
    result = verify(model, target)  # same bytes, same golden — or nothing is served
    ref = build_reference(
        model,
        train,
        validation,
        float(manifest["bands"]["block"]),
        cfg.review_budget_per_day,
        result.artifact_sha256,
    )
    save_reference(ref, target)
    manifest_path(target).write_text(json.dumps(manifest, indent=2))
    return target


def build_manifest(
    run_name: str,
    candidate_artifact: Path,
    metrics: dict[str, float],
    gates: list[GateResult],
    info: dict[str, Any],
    registry_version: int | None,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "model_version": f"{run_name}+{info.get('calibration', 'calibrated')}",
        "artifact_sha256": artifact_digest(candidate_artifact),
        "source_artifact": candidate_artifact.name,
        "registry_version": registry_version,
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bands": {"review": metrics["review_threshold"], "block": metrics["block_threshold"]},
        "metrics": metrics,
        "gates": [asdict(g) for g in gates],
        "model_info": {**info, "validation_pr_auc": metrics["pr_auc"]},
    }


# --- the registry --------------------------------------------------------------------


def register_candidate(
    tracking_uri: str,
    cfg: PromotionConfig,
    candidate_artifact: Path,
    run_name: str,
    gates: list[GateResult],
    promoted: bool,
) -> int:
    """One registry version per evaluated candidate; the champion alias moves on promotion."""
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        client.create_registered_model(cfg.registry_model)
    except MlflowException:
        pass  # exists
    verdict = "promoted" if promoted else "rejected" if not all_pass(gates) else "passed"
    version = client.create_model_version(
        cfg.registry_model,
        source=candidate_artifact.resolve().as_uri(),
        tags={
            "run_name": run_name,
            "artifact_sha256": artifact_digest(candidate_artifact),
            "verdict": verdict,
            "gates": json.dumps([asdict(g) for g in gates]),
        },
        description=f"{run_name}: {verdict}; "
        + ", ".join(f"{g.metric} {'ok' if g.passed else 'FAIL'}" for g in gates),
    )
    if promoted:
        client.set_registered_model_alias(cfg.registry_model, "champion", version.version)
    return int(version.version)


def champion_version(tracking_uri: str, registry_model: str) -> int | None:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri)
    try:
        return int(MlflowClient().get_model_version_by_alias(registry_model, "champion").version)
    except MlflowException:
        return None
