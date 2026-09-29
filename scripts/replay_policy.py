"""Price the served policy as the service applies it, beside the forms that used to stand in.

Writes reports/policy/replay.{csv,md}: one row per way of deciding actions on the
validation window, all at the champion's block threshold and the promotion budget,
and each at several review catch rates (the share of reviewed fraud an analyst stops).

    fixed band        promotion before item 9: a review threshold sized on validation
    pooled window     the monitoring reference before item 9: budget x days, one ranking
    daily upload      replay_served_policy, each day one request (what promotion gates on)
    hourly / 15 min   replay_served_policy, requests in time order (an online queue)

Validation only: the reporting window is not consulted (ADR 0002).

Example:
    uv run python scripts/replay_policy.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud.data import schema
from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.evaluate.policy import (
    Policy,
    apply_policy,
    apply_rank_policy,
    load_policy_config,
    policy_outcome,
    replay_served_policy,
    size_review_band,
)
from fraud.evaluate.threshold import load_threshold_config
from fraud.serve.parity import read_manifest, verify
from fraud.train.promotion import load_promotion_config

ROOT = Path(__file__).resolve().parents[1]
CATCH_RATES = (1.0, 0.9, 0.7)
KEEP = [
    "reviewed_per_day",
    "review_precision",
    "recall_block_plus_review",
    "fraud_amount_caught_share",
    "cost_per_transaction",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion", type=Path, default=ROOT / "models" / "champion" / "model.joblib"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "policy")
    args = parser.parse_args()

    model = joblib.load(args.champion)
    verify(model, args.champion)
    manifest = read_manifest(args.champion)
    if manifest is None:
        raise SystemExit(f"{args.champion} has no manifest: which block threshold does it serve?")
    block_t = float(manifest["bands"]["block"])
    budget = load_promotion_config(ROOT / "configs" / "promotion.yaml").review_budget_per_day
    tcfg = load_threshold_config(ROOT / "configs" / "threshold.yaml")
    pcfg = load_policy_config(ROOT / "configs" / "policy.yaml")

    parts = split(
        load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"),
        load_split_config(ROOT / "configs" / "split.yaml"),
    )
    v = parts["validation"]
    y = v[schema.TARGET_COL].to_numpy(dtype=int)
    p = np.asarray(model.predict_proba(v)[:, 1], dtype=float)
    amount = v["TransactionAmt"].to_numpy(dtype=float)
    dt = v[schema.TIME_COL].to_numpy()
    day = dt // 86_400
    n_days = len(np.unique(day))

    review_t = size_review_band(p, day, block_t, budget)
    policies = {
        "fixed band (old promotion)": apply_policy(p, Policy(block_t, review_t, budget)),
        "pooled window (old reference)": apply_rank_policy(p, block_t, budget * n_days)[0],
        "daily upload (served; gated)": replay_served_policy(p, dt, block_t, budget),
        "hourly requests (served)": replay_served_policy(p, dt, block_t, budget, 3600),
        "15-minute requests (served)": replay_served_policy(p, dt, block_t, budget, 900),
    }
    rows = []
    for name, action in policies.items():
        for c in CATCH_RATES:
            m = policy_outcome(y, action, amount, n_days, tcfg.costs, pcfg.review, c)
            m["cost_per_transaction"] = m["total_cost"] / len(y)
            rows.append({"policy": name, "review_catch_rate": c, **{k: m[k] for k in KEEP}})
    table = pd.DataFrame(rows)
    daily = policies["daily upload (served; gated)"]
    hourly = policies["hourly requests (served)"]
    agree = float((daily == hourly).mean())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "replay.csv", index=False)
    header = (
        f"Champion {manifest['model_version']} ({manifest['artifact_sha256'][:12]}), validation "
        f"days {int(day.min())}-{int(day.max())}, {len(y):,} transactions, block >= {block_t}, "
        f"review budget {budget}/day. Daily and hourly replays agree on {agree:.1%} of actions.\n\n"
    )
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table.itertuples(index=False):
        lines.append(
            "| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |"
        )
    (args.out_dir / "replay.md").write_text(header + "\n".join(lines) + "\n")
    print(header + table.to_string(index=False))


if __name__ == "__main__":
    main()
