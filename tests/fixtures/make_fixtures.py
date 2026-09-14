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
N_FRAUD = 14
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
    fraud_idx = rng.choice(N_TX, size=N_FRAUD, replace=False)
    frame[schema.TARGET_COL] = np.isin(np.arange(N_TX), fraud_idx).astype(np.int64)

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


def main() -> None:
    rng = np.random.default_rng(SEED)
    tx = make_transactions(rng)
    idn = make_identity(rng, tx[schema.ID_COL].to_numpy())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tx.to_csv(OUT_DIR / "train_transaction.csv", index=False)
    idn.to_csv(OUT_DIR / "train_identity.csv", index=False)
    print(f"wrote {len(tx)} transactions, {len(idn)} identity rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
