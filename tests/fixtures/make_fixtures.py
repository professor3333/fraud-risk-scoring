"""Generate small synthetic raw files with the IEEE-CIS schema.

The values are random and carry no signal; the fixture exists to exercise the
data contract (column names, dtypes, nullability, join cardinality), not to
train anything. Run ``uv run python tests/fixtures/make_fixtures.py`` to
regenerate; the output is committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud.data import schema

N_TX = 400
N_IDENTITY = 100
N_FRAUD_PER_WINDOW = {"train": 12, "validation": 5, "test": 5}
WINDOW_DAYS = {"train": (1, 122), "validation": (123, 152), "test": (153, 183)}
SEED = 20240914
OUT_DIR = Path(__file__).resolve().parent / "raw"


def make_transactions(rng: np.random.Generator) -> pd.DataFrame:
    ids = np.arange(2_987_000, 2_987_000 + N_TX)
    dt = np.sort(rng.integers(86_400, 86_400 * 184, size=N_TX))
    frame: dict[str, object] = {
        schema.ID_COL: ids,
        schema.TARGET_COL: np.zeros(N_TX, dtype=np.int64),
        schema.TIME_COL: dt,
        "TransactionAmt": np.round(rng.lognormal(4.0, 1.0, size=N_TX), 2),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=N_TX),
        "card1": rng.integers(1000, 18_000, size=N_TX),
    }
    # Positives in every split window, so window-level metrics are defined.
    day = dt // 86_400
    fraud_idx = np.concatenate(
        [
            rng.choice(np.flatnonzero((day >= lo) & (day <= hi)), size=n, replace=False)
            for name, n in N_FRAUD_PER_WINDOW.items()
            for lo, hi in [WINDOW_DAYS[name]]
        ]
    )
    is_fraud = np.isin(np.arange(N_TX), fraud_idx)
    frame[schema.TARGET_COL] = is_fraud.astype(np.int64)
    # Plant a weak, honest signal so model tests can require "beats the prior":
    # fraud rows skew towards ProductCD == "C" and larger amounts.
    product = np.asarray(frame["ProductCD"], dtype=object)
    product[is_fraud & (rng.random(N_TX) < 0.9)] = "C"
    frame["ProductCD"] = product
    amt = np.asarray(frame["TransactionAmt"], dtype=float)
    amt[is_fraud] = np.round(amt[is_fraud] * 10.0, 2)
    frame["TransactionAmt"] = amt

    def sparse_float(p_null: float, lo: float, hi: float) -> np.ndarray:
        vals = rng.uniform(lo, hi, size=N_TX).round(1)
        vals[rng.random(N_TX) < p_null] = np.nan
        return vals

    def sparse_str(choices: list[str], p_null: float) -> np.ndarray:
        vals = rng.choice(choices, size=N_TX).astype(object)
        vals[rng.random(N_TX) < p_null] = None
        return vals

    for c in ("card2", "card3", "card5"):
        frame[c] = sparse_float(0.05, 100, 600)
    frame["card4"] = sparse_str(["visa", "mastercard", "discover"], 0.02)
    frame["card6"] = sparse_str(["debit", "credit"], 0.02)
    frame["addr1"] = sparse_float(0.1, 100, 500)
    frame["addr2"] = sparse_float(0.1, 10, 100)
    frame["dist1"] = sparse_float(0.6, 0, 3000)
    frame["dist2"] = sparse_float(0.9, 0, 3000)
    frame["P_emaildomain"] = sparse_str(["gmail.com", "yahoo.com", "hotmail.com"], 0.15)
    frame["R_emaildomain"] = sparse_str(["gmail.com", "yahoo.com", "anonymous.com"], 0.75)
    for c in schema.C_COLS:
        frame[c] = rng.integers(0, 20, size=N_TX).astype(float)
    for c in schema.D_COLS:
        frame[c] = sparse_float(0.5, 0, 600)
    for c in schema.M_COLS:
        frame[c] = sparse_str(["T", "F"], 0.4)
    frame["M4"] = sparse_str(["M0", "M1", "M2"], 0.4)
    for c in schema.V_COLS:
        frame[c] = sparse_float(0.7, 0, 5)
    return pd.DataFrame(frame)[list(schema.TRANSACTION_COLS)]


def make_identity(rng: np.random.Generator, tx_ids: np.ndarray) -> pd.DataFrame:
    ids = np.sort(rng.choice(tx_ids, size=N_IDENTITY, replace=False))
    frame: dict[str, object] = {schema.ID_COL: ids}
    for col in schema.IDENTITY_COLS[1:]:
        if col in schema.IDENTITY_STR_COLS:
            vals = rng.choice(["Found", "NotFound", "New"], size=N_IDENTITY).astype(object)
            vals[rng.random(N_IDENTITY) < 0.3] = None
        else:
            vals = rng.uniform(-50, 50, size=N_IDENTITY).round(1)
            vals[rng.random(N_IDENTITY) < 0.3] = np.nan
        frame[col] = vals
    frame["DeviceType"] = rng.choice(["desktop", "mobile"], size=N_IDENTITY)
    return pd.DataFrame(frame)[list(schema.IDENTITY_COLS)]


def write_golden() -> None:
    """Fit the fixture pipeline from the committed config and store the preprocessed
    feature matrix for a frozen sample.

    The matrix (not the model output) is the golden: preprocessing is deterministic
    across platforms, whereas XGBoost trees fitted on a tiny sample are not. Model
    outputs are pinned per artifact by scripts/freeze_artifact.py instead.
    """
    import json
    import math

    from fraud.data.load import load_train
    from fraud.data.split import load_split_config, split
    from fraud.features.columns import load_feature_spec
    from fraud.serve.parity import choose_sample
    from fraud.train.run import fit_and_evaluate, load_train_config

    root = Path(__file__).resolve().parents[2]
    cfg = load_train_config(root / "configs" / "model" / "xgboost.yaml")
    spec = load_feature_spec(cfg.features)
    parts = split(load_train(OUT_DIR), load_split_config(root / "configs" / "split.yaml"))
    pipe, _, _ = fit_and_evaluate(parts, spec, cfg)
    sample = choose_sample(parts["test"], n_per_group=5)
    matrix = pipe[:-1].transform(sample)
    golden = {
        "feature_names": list(pipe.named_steps["features"].get_feature_names_out()),
        "rows": {
            str(int(i)): [None if math.isnan(v) else round(float(v), 9) for v in row]
            for i, row in zip(sample[schema.ID_COL], matrix, strict=True)
        },
    }
    (OUT_DIR.parent / "frozen_features.json").write_text(json.dumps(golden))
    print(f"wrote golden feature matrix: {len(golden['rows'])} rows x {matrix.shape[1]} features")


def main() -> None:
    import sys

    if "--golden" in sys.argv:
        write_golden()
        return
    rng = np.random.default_rng(SEED)
    tx = make_transactions(rng)
    idn = make_identity(rng, tx[schema.ID_COL].to_numpy())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tx.to_csv(OUT_DIR / "train_transaction.csv", index=False)
    idn.to_csv(OUT_DIR / "train_identity.csv", index=False)
    print(f"wrote {len(tx)} transactions, {len(idn)} identity rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
