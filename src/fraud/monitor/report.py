"""Monitoring report from stored predictions against the frozen reference.

Four sections, as in the design: API (rate, latency, errors), predictions (score
distribution, action shares), data (input distributions, missingness), model
(score / feature drift and — when labels are supplied — eventual PR-AUC,
precision, recall, calibration). Drift is measured with the population
stability index; >= 0.10 warns, >= 0.20 alerts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from fraud.evaluate.calibration import calibration_metrics
from fraud.evaluate.metrics import compute_metrics
from fraud.monitor.drift import (
    binned_shares,
    categorical_shares,
    level,
    numeric_summary,
    psi,
)
from fraud.monitor.feedback import attach_outcomes, early_section, label_section
from fraud.monitor.reference import BINARY, CATEGORICAL, NUMERIC


def _window_seconds(started: pd.Series) -> float:
    if started.empty:
        return 0.0
    t = pd.to_datetime(started, utc=True)
    return max(float((t.max() - t.min()).total_seconds()), 1.0)


def api_section(requests: pd.DataFrame) -> dict[str, Any]:
    if requests.empty:
        return {"requests": 0}
    seconds = _window_seconds(requests["started_at"])
    errors = requests[requests["status_code"] >= 400]
    by_endpoint = {
        ep: {
            "requests": int(len(g)),
            "rows": int(g["n_rows"].sum()),
            "latency_p50_ms": float(g["latency_ms"].quantile(0.5)),
            "latency_p95_ms": float(g["latency_ms"].quantile(0.95)),
            "errors": int((g["status_code"] >= 400).sum()),
        }
        for ep, g in requests.groupby("endpoint")
    }
    return {
        "requests": int(len(requests)),
        "rows_scored": int(requests["n_rows"].sum()),
        "window_seconds": seconds,
        "requests_per_second": float(len(requests) / seconds),
        "latency_p50_ms": float(requests["latency_ms"].quantile(0.5)),
        "latency_p95_ms": float(requests["latency_ms"].quantile(0.95)),
        "latency_max_ms": float(requests["latency_ms"].max()),
        "errors": int(len(errors)),
        "error_rate": float(len(errors) / len(requests)),
        "by_endpoint": by_endpoint,
    }


def predictions_section(events: pd.DataFrame, ref: dict[str, Any]) -> dict[str, Any]:
    if events.empty:
        return {"rows": 0}
    p = events["fraud_probability"]
    bins = binned_shares(p, ref["scores"]["edges"])
    actions = {a: float((events["action"] == a).mean()) for a in ("block", "review", "approve")}
    score_psi = psi(ref["scores"]["bins"], bins)
    action_psi = psi(ref["actions"], actions)
    return {
        "rows": int(len(events)),
        "score": {**numeric_summary(p), "reference_mean": ref["scores"]["mean"]},
        "score_bins": bins,
        "score_psi": score_psi,
        "score_drift": level(score_psi),
        "actions": actions,
        "reference_actions": ref["actions"],
        "action_psi": action_psi,
        "action_drift": level(action_psi),
        "policies": events["policy"].value_counts().to_dict(),
        "model_versions": events["model_version"].value_counts().to_dict(),
    }


def data_section(inputs: pd.DataFrame, ref: dict[str, Any]) -> dict[str, Any]:
    if inputs.empty:
        return {"rows": 0}
    out: dict[str, Any] = {"rows": int(len(inputs)), "features": {}}
    worst = 0.0
    for col in CATEGORICAL:
        shares = categorical_shares(inputs[col], keys=set(ref["features"][col]))
        value = psi(ref["features"][col], shares)
        worst = max(worst, value)
        out["features"][col] = {"psi": value, "drift": level(value), "shares": shares}
    for col in BINARY:
        rate = float(inputs[col].mean())
        r = float(ref["features"][col]["rate"])
        value = psi({"1": r, "0": 1 - r}, {"1": rate, "0": 1 - rate})
        worst = max(worst, value)
        out["features"][col] = {
            "rate": rate,
            "reference_rate": r,
            "psi": value,
            "drift": level(value),
        }
    for col in NUMERIC:
        bins = binned_shares(inputs[col], ref["numeric_edges"][col])
        value = psi(ref["features"][col]["bins"], bins)
        worst = max(worst, value)
        out["features"][col] = {
            **numeric_summary(inputs[col]),
            "reference_mean": ref["features"][col].get("mean"),
            "psi": value,
            "drift": level(value),
        }
    out["worst_psi"] = worst
    out["drift"] = level(worst)
    return out


def _evaluable(joined: pd.DataFrame) -> bool:
    """Ranking and calibration metrics need both classes present."""
    return len(joined) > 0 and joined["is_fraud"].nunique() == 2


def _performance(joined: pd.DataFrame, ref: dict[str, Any]) -> dict[str, Any]:
    y = joined["is_fraud"].to_numpy(dtype=int)
    p = joined["fraud_probability"].to_numpy(dtype=float)
    blocked = joined["action"] == "block"
    flagged = joined["action"] != "approve"
    m = compute_metrics(y, p, float(joined["block_threshold"].iloc[0]))
    cal = calibration_metrics(y, p)
    perf = {
        "n_labelled": int(len(joined)),
        "positive_rate": float(y.mean()),
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "block_precision": float(y[blocked].mean()) if blocked.any() else None,
        "recall_block": float(y[blocked].sum() / max(y.sum(), 1)),
        "recall_block_plus_review": float(y[flagged].sum() / max(y.sum(), 1)),
        "cal_brier": cal["brier"],
        "cal_ece": cal["ece"],
    }
    r = ref["performance"]
    perf["delta_pr_auc_vs_reference"] = perf["pr_auc"] - r["pr_auc"]
    perf["delta_block_precision_vs_reference"] = (
        None if perf["block_precision"] is None else perf["block_precision"] - r["block_precision"]
    )
    return perf


def model_section(
    events: pd.DataFrame,
    ref: dict[str, Any],
    outcomes: pd.DataFrame | None,
    clock: int | None = None,
    maturity_days: int = 0,
) -> dict[str, Any]:
    """Eventual performance from outcomes joined onto events.

    `clock` is the TransactionDT time the label feed has reached. With it, eventual
    metrics use only cohorts whose reporting window has closed and the open cohorts
    contribute an early signal (fraud.monitor.feedback). Without it the outcomes are
    taken as complete and final (a fully matured label file).
    """
    out: dict[str, Any] = {
        "reference_performance": ref["performance"],
        "labels": None,
        "early": None,
    }
    if outcomes is None or events.empty:
        out["eventual"] = None
        return out
    if clock is None:
        joined = events.merge(
            outcomes[["transaction_id", "is_fraud"]].drop_duplicates("transaction_id"),
            on="transaction_id",
            how="inner",
        )
        out["eventual"] = (
            _performance(joined, ref) if _evaluable(joined) else {"n_labelled": len(joined)}
        )
        return out
    attached = attach_outcomes(events, outcomes, clock, maturity_days)
    out["labels"] = label_section(attached, clock, maturity_days)
    out["early"] = early_section(attached)
    closed = attached.loc[attached["cohort_closed"] & (attached["label_status"] == "matured")]
    out["eventual"] = (
        _performance(closed, ref) if _evaluable(closed) else {"n_labelled": len(closed)}
    )
    return out


def _flags(report: dict[str, Any]) -> list[str]:
    pred: dict[str, Any] = report["predictions"]
    data: dict[str, Any] = report["data"]
    api: dict[str, Any] = report["api"]
    ev: dict[str, Any] = report["model"].get("eventual") or {}
    flags: list[str] = []
    if pred.get("score_drift") in ("warn", "alert"):
        flags.append(f"score drift {pred['score_drift']} (PSI {pred['score_psi']:.3f})")
    if pred.get("action_drift") in ("warn", "alert"):
        flags.append(f"action-share drift {pred['action_drift']} (PSI {pred['action_psi']:.3f})")
    for col, f in data.get("features", {}).items():
        if f["drift"] in ("warn", "alert"):
            flags.append(f"feature drift {f['drift']}: {col} (PSI {f['psi']:.3f})")
    if api.get("error_rate", 0) > 0.05:
        flags.append(f"error rate {api['error_rate']:.1%}")
    d_pr = ev.get("delta_pr_auc_vs_reference")
    if d_pr is not None and d_pr < -0.03:
        flags.append(f"eventual PR-AUC {ev['pr_auc']:.3f} is {d_pr:+.3f} vs reference")
    d_bp = ev.get("delta_block_precision_vs_reference")
    if d_bp is not None and d_bp < -0.05:
        flags.append(f"block precision {ev['block_precision']:.2f} is {d_bp:+.2f} vs reference")
    labels: dict[str, Any] = report["model"].get("labels") or {}
    if labels.get("overdue"):
        flags.append(
            f"label feed gap: {labels['overdue']} transactions past the"
            f" {labels['maturity_days']}-day window without an outcome"
        )
    return flags


def build_report(
    requests: pd.DataFrame,
    events: pd.DataFrame,
    inputs: pd.DataFrame,
    ref: dict[str, Any],
    outcomes: pd.DataFrame | None = None,
    window: tuple[str | None, str | None] = (None, None),
    clock: int | None = None,
    maturity_days: int = 0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "window": {"since": window[0], "until": window[1]},
        "api": api_section(requests),
        "predictions": predictions_section(events, ref),
        "data": data_section(inputs, ref),
        "model": model_section(events, ref, outcomes, clock, maturity_days),
    }
    flags = _flags(report)
    serious = ("alert", "eventual", "block precision", "error rate", "label feed gap")
    report["flags"] = flags
    report["status"] = (
        "alert" if any(any(k in f for k in serious) for f in flags) else "warn" if flags else "ok"
    )
    return report


def _api_lines(api: dict[str, Any]) -> list[str]:
    if not api.get("requests"):
        return ["- no requests in the window"]
    lines = [
        f"- requests {api['requests']} ({api['requests_per_second']:.3f}/s over "
        f"{api['window_seconds']:.0f} s), rows scored {api['rows_scored']}",
        f"- latency p50 {api['latency_p50_ms']:.0f} ms · p95 {api['latency_p95_ms']:.0f} ms"
        f" · max {api['latency_max_ms']:.0f} ms",
        f"- errors {api['errors']} ({api['error_rate']:.1%})",
        "",
        "| endpoint | requests | rows | p50 ms | p95 ms | errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ep, g in api["by_endpoint"].items():
        lines.append(
            f"| `{ep}` | {g['requests']} | {g['rows']} | {g['latency_p50_ms']:.0f} | "
            f"{g['latency_p95_ms']:.0f} | {g['errors']} |"
        )
    return lines


def _prediction_lines(pred: dict[str, Any]) -> list[str]:
    if not pred.get("rows"):
        return ["- no predictions in the window"]
    s, a, r = pred["score"], pred["actions"], pred["reference_actions"]
    return [
        f"- rows {pred['rows']} · mean probability {s['mean']:.4f} "
        f"(reference {s['reference_mean']:.4f}) · p50 {s['p50']:.4f} · p99 {s['p99']:.4f}",
        f"- score PSI {pred['score_psi']:.3f} → **{pred['score_drift']}**",
        f"- actions: block {a['block']:.1%} · review {a['review']:.1%} · approve {a['approve']:.1%}"
        f" (reference {r['block']:.1%} / {r['review']:.1%} / {r['approve']:.1%});"
        f" PSI {pred['action_psi']:.3f} → **{pred['action_drift']}**",
        f"- model versions: {pred['model_versions']}; policies: {pred['policies']}",
    ]


def _data_lines(data: dict[str, Any]) -> list[str]:
    if not data.get("rows"):
        return ["- no input snapshots in the window"]
    lines = ["| feature | PSI | drift | now vs reference |", "|---|---:|---|---|"]
    for col, f in data["features"].items():
        if "rate" in f:
            detail = f"rate {f['rate']:.3f} vs {f['reference_rate']:.3f}"
        elif "mean" in f:
            detail = f"mean {f['mean']:.2f} vs {f['reference_mean']:.2f}"
        else:
            top = sorted(f["shares"].items(), key=lambda kv: -kv[1])[:3]
            detail = ", ".join(f"{k} {v:.1%}" for k, v in top)
        lines.append(f"| {col} | {f['psi']:.3f} | {f['drift']} | {detail} |")
    return lines


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _model_lines(model: dict[str, Any]) -> list[str]:
    r = model["reference_performance"]
    lines = [
        f"- reference (validation): PR-AUC {r['pr_auc']:.3f} · ROC-AUC {r['roc_auc']:.3f}"
        f" · block precision {r['block_precision']:.2f} · Brier {r['cal_brier']:.4f}"
        f" · ECE {r['cal_ece']:.4f}"
    ]
    lab = model.get("labels")
    if lab:
        lines += [
            f"- labels as of day {lab['as_of_day']} ({lab['maturity_days']}-day window):"
            f" {lab['matured']} matured · {lab['pending']} pending · {lab['overdue']} overdue"
            + (f" · {lab['unknown_time']} without a time" if lab["unknown_time"] else "")
            + f"; closed cohort {lab['closed_cohort']} ({_pct(lab['closed_share'])} of the window)",
            f"- positive rate: closed cohort {_pct(lab['positive_rate_closed'])} vs labels that"
            f" have arrived {_pct(lab['positive_rate_arrived'])} (the latter is biased while"
            " cohorts are open)",
        ]
        if lab["report_lag_days_median"] is not None:
            lines.append(
                f"- report lag of arrived fraud: median {lab['report_lag_days_median']:.0f} d"
                f" · p90 {lab['report_lag_days_p90']:.0f} d"
            )
    early = model.get("early")
    if early and early["fraud_reported_so_far"]:
        lines.append(
            f"- early signal on {early['open_cohort']} open rows: {early['fraud_reported_so_far']}"
            f" fraud reported so far ({early['reported_rate_so_far']:.2%} and rising);"
            f" of those blocked {early['early_recall_block']:.2f}, blocked or reviewed"
            f" {early['early_recall_block_plus_review']:.2f}"
        )
    ev = model.get("eventual")
    if ev is None:
        lines.append(
            "- eventual performance: no outcomes (run scripts/feedback.py, POST /outcomes,"
            " or pass --labels)"
        )
    elif "pr_auc" not in ev:
        lines.append(
            f"- eventual performance: {ev['n_labelled']} labelled in closed cohorts, too few"
            " (or one class only) to evaluate"
            if lab
            else f"- eventual performance: {ev['n_labelled']} labelled, too few to evaluate"
        )
    else:
        lines += [
            f"- eventual ({ev['n_labelled']} labelled, positive rate {ev['positive_rate']:.3f}):"
            f" PR-AUC {ev['pr_auc']:.3f} ({ev['delta_pr_auc_vs_reference']:+.3f})"
            f" · ROC-AUC {ev['roc_auc']:.3f}",
            f"- block precision {ev['block_precision']:.2f}"
            f" ({ev['delta_block_precision_vs_reference']:+.2f}) · recall block"
            f" {ev['recall_block']:.2f} · recall block+review {ev['recall_block_plus_review']:.2f}",
            f"- calibration: Brier {ev['cal_brier']:.4f} · ECE {ev['cal_ece']:.4f}",
        ]
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    w = report["window"]
    span = f"{w['since'] or 'start'} → {w['until'] or 'now'}"
    lines = [
        f"# Monitoring report — {report['status'].upper()}",
        "",
        f"Generated {report['generated_at']}; window {span}.",
        "",
        "## Flags",
        "",
        *([f"- {f}" for f in report["flags"]] or ["- none"]),
        "",
        "## API",
        "",
        *_api_lines(report["api"]),
        "",
        "## Predictions",
        "",
        *_prediction_lines(report["predictions"]),
        "",
        "## Data",
        "",
        *_data_lines(report["data"]),
        "",
        "## Model",
        "",
        *_model_lines(report["model"]),
    ]
    return "\n".join(lines) + "\n"
