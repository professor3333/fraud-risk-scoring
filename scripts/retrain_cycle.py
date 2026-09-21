"""One scheduled retraining cycle: decide, retrain, gate, promote, write the patch (ADR 0012).

    label feed clock (+ the monitor's report)  ->  should a cycle run at all?
      -> challenger fit on days <= train_end, calibrated out-of-fold inside that range
      -> challenger AND the serving champion scored on the month neither was fit on
      -> ADR 0010's acceptance gates + a margin an unattended job must clear
      -> promote: models/champion/ replaced, registry alias moved
      -> write configs/serving.yaml's bands and champion_sha256 for review

It stops there. Publishing the artifact and pointing the host at it are the
workflow's job (.github/workflows/retrain.yml), and the policy edit this writes is
merged by a human — the one step that is not automated, and the right one.

Example:
    uv run python scripts/retrain_cycle.py --dry-run
    uv run python scripts/retrain_cycle.py --as-of-day 152 --label-maturity-days 0
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import yaml

from fraud.data.load import load_train
from fraud.evaluate.policy import load_policy_config
from fraud.evaluate.threshold import load_threshold_config
from fraud.serve.parity import read_manifest
from fraud.train.cycle import CycleResult, cycle_info, month_frame, run_cycle, training_frame
from fraud.train.lifecycle import load_retrain_config
from fraud.train.promotion import (
    CHAMPION_ARTIFACT,
    build_manifest,
    load_promotion_config,
    materialise,
    register_candidate,
)
from fraud.train.serving_config import update_serving_config
from fraud.train.trigger import TriggerDecision, TriggerState, decide, load_cycle_policy

ROOT = Path(__file__).resolve().parents[1]


def champion_state(champion_dir: Path) -> tuple[Any | None, dict[str, Any] | None]:
    """The serving champion and its manifest, or (None, None) before the first promotion."""
    artifact = champion_dir / CHAMPION_ARTIFACT
    if not artifact.exists():
        return None, None
    manifest = read_manifest(artifact)
    if manifest is None:
        return None, None  # a stand-in without a manifest is not a champion to compare with
    return joblib.load(artifact), manifest


def trained_through(manifest: dict[str, Any] | None) -> int | None:
    if manifest is None:
        return None
    window = manifest.get("model_info", {}).get("training_window_days")
    return None if not window else int(window[1])


def feed_clock_day(audit_db: Path) -> int | None:
    """The TransactionDT day the label feed has advanced to, or None."""
    if not audit_db.exists():
        return None
    from fraud.monitor.feedback import SECONDS_PER_DAY
    from fraud.serve.audit import AuditLog

    clock = AuditLog(audit_db).feed_clock()
    return None if clock is None else int(clock // SECONDS_PER_DAY)


def monitor_delta(path: Path | None) -> float | None:
    """`delta_pr_auc_vs_reference` from a monitoring report, when one was passed."""
    if path is None:
        return None
    report = json.loads(path.read_text())
    eventual = (report.get("model") or {}).get("eventual") or {}
    value = eventual.get("delta_pr_auc_vs_reference")
    return None if value is None else float(value)


def log_to_mlflow(
    tracking_uri: str,
    result: CycleResult,
    decision: TriggerDecision,
    policy: Any,
    retrain: Any,
    report: str,
    promoted: bool,
) -> None:
    """One MLflow run per cycle, promoted or not (§8: a run that is not logged did not
    happen). The registry records the candidate; this records how it was judged."""
    import mlflow

    from fraud.train.run import git_commit

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-retrain-cycle")
    with mlflow.start_run(run_name=f"cycle_day{result.as_of_day}"):
        mlflow.log_params(
            {
                "git_commit": git_commit(),
                "as_of_day": result.as_of_day,
                "trigger_rule": decision.rule,
                "label_maturity_days": policy.label_maturity_days,
                "promotion_margin": policy.promotion_margin,
                "model_config": retrain.model_config.name,
                "train_end": result.train_end,
                "evaluation_month": f"{result.month[0]}-{result.month[1]}",
                "champion": result.champion_run_name or "none",
                "champion_trained_through": result.champion_trained_through,
                "promoted": promoted,
            }
        )
        mlflow.log_metrics(
            {
                **{f"challenger_{k}": v for k, v in result.challenger_metrics.items()},
                **{f"champion_{k}": v for k, v in (result.champion_metrics or {}).items()},
                **({} if result.delta_pr_auc is None else {"delta_pr_auc": result.delta_pr_auc}),
                "gates_passed": float(result.gates_passed),
                "margin_ok": float(result.margin_ok),
            }
        )
        mlflow.log_text(report, f"cycle_day{result.as_of_day}.md")


def github_output(**values: Any) -> None:
    """Hand the workflow what it would otherwise have to re-derive from the file system
    (which report, which digest, whether anything was promoted). A no-op off CI."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def _verdict(result: CycleResult, promoted: bool) -> str:
    if promoted:
        return "PROMOTE"
    # A bootstrap that fails a gate has nothing to fall back on: saying "keep the
    # champion" would describe a champion that does not exist.
    return "KEEP THE CHAMPION" if result.champion_run_name else "REJECT (no champion exists)"


def render(result: CycleResult, decision: TriggerDecision, promoted: bool) -> str:
    lines = [
        f"# Retraining cycle — as of day {result.as_of_day}",
        "",
        f"Trigger: **{decision.rule}** — {decision.reason}",
        "",
        f"- labels mature through day {result.mature_through}; challenger trained on days "
        f"1–{result.train_end}",
        f"- evaluation month days {result.month[0]}–{result.month[1]}: {result.n_month:,} rows, "
        f"{result.positives:,} fraud (neither model was fit on it)",
        f"- champion: {result.champion_run_name or 'none (bootstrap)'}"
        + (
            f", trained through day {result.champion_trained_through}"
            if result.champion_trained_through is not None
            else ""
        ),
        "",
        "| metric | challenger | champion | gate |",
        "|---|---:|---:|---|",
    ]
    for gate in result.gates:
        champion = "—" if gate.champion is None else f"{gate.champion:.4f}"
        verdict = "PASS" if gate.passed else "**FAIL**"
        lines.append(
            f"| {gate.metric} | {gate.candidate:.4f} | {champion} | {verdict} — {gate.reason} |"
        )
    delta = result.delta_pr_auc
    lines += [
        "",
        f"PR-AUC on the month: **{result.challenger_metrics['pr_auc']:.4f}**"
        + (
            f" vs champion {result.champion_metrics['pr_auc']:.4f} ({delta:+.4f})"
            if result.champion_metrics is not None and delta is not None
            else ""
        ),
        "- bands re-derived on this month: review "
        f"{result.challenger_metrics['review_threshold']:.4f} / block "
        f"{result.challenger_metrics['block_threshold']:.4f}",
        "",
        f"**Decision: {_verdict(result, promoted)}** — {result.reason}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "retrain.yaml")
    parser.add_argument("--promotion", type=Path, default=ROOT / "configs" / "promotion.yaml")
    parser.add_argument("--threshold", type=Path, default=ROOT / "configs" / "threshold.yaml")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs" / "policy.yaml")
    parser.add_argument("--serving-config", type=Path, default=ROOT / "configs" / "serving.yaml")
    parser.add_argument("--model-config", type=Path, default=None, help="challenger recipe")
    parser.add_argument(
        "--as-of-day", type=int, default=None,
        help="calendar day of the cycle; default: the label feed's clock",
    )  # fmt: skip
    parser.add_argument("--label-maturity-days", type=int, default=None)
    parser.add_argument(
        "--audit-db", type=Path, default=None, help="default: serving.yaml audit_db"
    )
    parser.add_argument(
        "--monitor-report", type=Path, default=None,
        help="a monitoring report JSON; its eventual PR-AUC delta can bring a cycle forward",
    )  # fmt: skip
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models" / "candidates")
    parser.add_argument("--champion-dir", type=Path, default=None, help="default: promotion.yaml")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "retrain" / "cycles")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="skip the calendar and monitor rules; the label and reporting-window guards"
        " still apply",
    )  # fmt: skip
    parser.add_argument(
        "--dry-run", action="store_true",
        help="train, score and gate; never promote, never touch configs/serving.yaml",
    )  # fmt: skip
    args = parser.parse_args()

    retrain = load_retrain_config(args.config)
    if args.model_config is not None:
        retrain = replace(retrain, model_config=args.model_config)
    policy = load_cycle_policy(args.config)
    if args.label_maturity_days is not None:
        policy = replace(policy, label_maturity_days=args.label_maturity_days)
    promotion = load_promotion_config(args.promotion)
    champion_dir = args.champion_dir or ROOT / promotion.champion_dir
    promotion = replace(promotion, champion_dir=champion_dir)

    champion, manifest = champion_state(champion_dir)
    serving = yaml.safe_load(args.serving_config.read_text())
    audit_db = args.audit_db or ROOT / serving["audit_db"]
    state = TriggerState(
        feed_clock_day=args.as_of_day if args.as_of_day is not None else feed_clock_day(audit_db),
        champion_trained_through=trained_through(manifest),
        monitor_pr_auc_delta=monitor_delta(args.monitor_report),
        forced=args.force,
    )
    decision = decide(state, policy)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    print(f"trigger: {'RUN' if decision.run else 'WAIT'} ({decision.rule}) — {decision.reason}")
    # No `or args.force` here: --force is an input to the decision, not an override of
    # it. The trigger's hard refusals — no matured labels, and the reporting window —
    # must not be bypassable from a command line or a workflow input (ADR 0002).
    if not decision.run:
        (args.report_dir / "last_decision.json").write_text(
            json.dumps({"trigger": asdict(decision), "cycle": None}, indent=2)
        )
        github_output(promoted="false", trigger=decision.rule)
        return
    assert decision.as_of_day is not None

    df = load_train(args.raw_dir, cache_dir=args.cache_dir)
    result = run_cycle(
        df, retrain, policy, promotion,
        load_threshold_config(args.threshold), load_policy_config(args.policy),
        decision.as_of_day, args.out_dir, champion,
        None if manifest is None else str(manifest["run_name"]),
        trained_through(manifest), args.seed,
    )  # fmt: skip
    promote = result.promote and not args.dry_run
    report = render(result, decision, promote)
    print(report)

    log_to_mlflow(args.tracking_uri, result, decision, policy, retrain, report, promote)
    version = register_candidate(
        args.tracking_uri, promotion, result.challenger_artifact,
        result.challenger_run_name, result.gates, promote,
    )  # fmt: skip
    info = cycle_info(
        result, retrain, calibration="sigmoid",
        experiment=f"retrain-cycle-day{result.as_of_day}",
    )  # fmt: skip
    new_manifest = build_manifest(
        result.challenger_run_name, result.challenger_artifact,
        result.challenger_metrics, result.gates, info, version,
    )  # fmt: skip
    stem = f"cycle_day{result.as_of_day}"
    (args.report_dir / f"{stem}.md").write_text(report)
    (args.report_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "trigger": asdict(decision),
                "cycle": {
                    **{k: v for k, v in asdict(result).items() if k != "challenger_artifact"},
                    "challenger_artifact": str(result.challenger_artifact),
                    "delta_pr_auc": result.delta_pr_auc,
                },
                "registry_version": version,
                "promoted": promote,
                "manifest": new_manifest,
            },
            indent=2,
            default=str,
        )
    )
    github_output(
        promoted=str(promote).lower(),
        trigger=decision.rule,
        report_md=str(args.report_dir / f"{stem}.md"),
        champion_sha=new_manifest["artifact_sha256"][:12],
    )
    if not promote:
        kept = (
            f"champion {result.champion_run_name} kept"
            if result.champion_run_name
            else "no champion promoted; nothing is serving from this cycle"
        )
        print(f"{kept} (registry version {version})")
        return

    challenger = joblib.load(result.challenger_artifact)
    # The monitoring reference follows the artifact: feature distributions from the rows
    # it was fit on, scores and policy from the month it was measured on (ADR 0011).
    materialise(
        result.challenger_artifact, challenger, promotion, new_manifest,
        training_frame(df, result.train_end), month_frame(df, result.month),
    )  # fmt: skip
    print(f"PROMOTED {result.challenger_run_name} -> {champion_dir} (registry version {version})")
    args.serving_config.write_text(
        update_serving_config(args.serving_config.read_text(), new_manifest)
    )
    print(
        f"wrote {args.serving_config.name}: bands review "
        f"{new_manifest['bands']['review']:.4f} / block {new_manifest['bands']['block']:.4f}, "
        f"champion_sha256 {new_manifest['artifact_sha256'][:12]} — review and merge that diff "
        "before the host is pointed at this champion"
    )


if __name__ == "__main__":
    main()
