"""Exploratory data analysis for the IEEE-CIS training data.

Reads the joined training frame, writes ``reports/eda/dataset_summary.json``,
``missingness.csv``, ``categorical_cardinality.csv`` and figures ``*.png``. The findings
are written up by hand in ``docs/eda.md``; this script produces the evidence.

Nothing here is fit or reused by the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraud.data import schema  # noqa: E402
from fraud.data.load import load_train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "eda"
SECONDS_PER_DAY = 86_400

LEGIT = "#2a78d6"
FRAUD = "#eb6834"
GRID = "#d9d8d3"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "font.size": 9,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    day = df[schema.TIME_COL] // SECONDS_PER_DAY
    return df.assign(
        day=day,
        week=day // 7,
        hour=(df[schema.TIME_COL] % SECONDS_PER_DAY) // 3600,
        dow=day % 7,
    )


# --- sections -------------------------------------------------------------------


def label_and_time(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    y = df[schema.TARGET_COL]
    stats["n_rows"] = int(len(df))
    stats["n_fraud"] = int(y.sum())
    stats["fraud_rate"] = float(y.mean())
    stats["dt_min_days"] = float(df[schema.TIME_COL].min() / SECONDS_PER_DAY)
    stats["dt_max_days"] = float(df[schema.TIME_COL].max() / SECONDS_PER_DAY)
    stats["dt_is_sorted"] = bool(df[schema.TIME_COL].is_monotonic_increasing)

    weekly = df.groupby("week")[schema.TARGET_COL].agg(["mean", "size"])
    stats["weekly_fraud_rate"] = {int(k): float(v) for k, v in weekly["mean"].items()}
    stats["weekly_volume"] = {int(k): int(v) for k, v in weekly["size"].items()}

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(weekly.index, weekly["size"], color=LEGIT, lw=2)
    axes[0].set_ylabel("transactions / week")
    axes[0].set_title("Volume and fraud rate by week")
    axes[1].plot(weekly.index, weekly["mean"] * 100, color=FRAUD, lw=2)
    axes[1].set_ylabel("fraud rate (%)")
    axes[1].set_xlabel("week since data start")
    save(fig, "01_weekly_volume_and_fraud_rate")

    daily = df.groupby("day").size()
    stats["daily_volume_min"] = int(daily.min())
    stats["daily_volume_max"] = int(daily.max())
    stats["n_days"] = int(daily.index.nunique())
    stats["days_with_zero_rows"] = int(daily.index.max() - daily.index.min() + 1 - len(daily))

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(daily.index, daily.values, color=LEGIT, lw=1)
    ax.set_xlabel("day since data start")
    ax.set_ylabel("transactions / day")
    ax.set_title("Daily volume")
    save(fig, "02_daily_volume")

    by_hour = df.groupby("hour")[schema.TARGET_COL].agg(["mean", "size"])
    by_dow = df.groupby("dow")[schema.TARGET_COL].agg(["mean", "size"])
    stats["hourly_fraud_rate"] = {int(k): float(v) for k, v in by_hour["mean"].items()}
    stats["hourly_volume"] = {int(k): int(v) for k, v in by_hour["size"].items()}
    stats["dow_volume"] = {int(k): int(v) for k, v in by_dow["size"].items()}
    stats["dow_fraud_rate"] = {int(k): float(v) for k, v in by_dow["mean"].items()}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].bar(by_hour.index, by_hour["size"], color=LEGIT, width=0.8)
    axes[0].set_xlabel("hour of day (DT mod 86400)")
    axes[0].set_ylabel("transactions")
    axes[0].set_title("Volume by hour")
    axes[1].bar(by_hour.index, by_hour["mean"] * 100, color=FRAUD, width=0.8)
    axes[1].set_xlabel("hour of day (DT mod 86400)")
    axes[1].set_ylabel("fraud rate (%)")
    axes[1].set_title("Fraud rate by hour")
    save(fig, "03_hour_of_day")


def identity_coverage(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    has = df[schema.HAS_IDENTITY_COL]
    y = df[schema.TARGET_COL]
    stats["identity_coverage"] = float(has.mean())
    stats["fraud_rate_with_identity"] = float(y[has].mean())
    stats["fraud_rate_without_identity"] = float(y[~has].mean())
    weekly = df.groupby("week").agg(
        coverage=(schema.HAS_IDENTITY_COL, "mean"),
    )
    weekly_by_id = df.groupby(["week", schema.HAS_IDENTITY_COL])[schema.TARGET_COL].mean().unstack()
    stats["weekly_identity_coverage"] = {int(k): float(v) for k, v in weekly["coverage"].items()}
    stats["identity_by_product"] = {
        str(k): float(v) for k, v in df.groupby("ProductCD")[schema.HAS_IDENTITY_COL].mean().items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(weekly.index, weekly["coverage"] * 100, color=LEGIT, lw=2)
    axes[0].set_ylabel("rows with identity (%)")
    axes[0].set_xlabel("week")
    axes[0].set_title("Identity coverage over time")
    axes[1].plot(
        weekly_by_id.index, weekly_by_id[True] * 100, color=FRAUD, lw=2, label="with identity"
    )
    axes[1].plot(
        weekly_by_id.index, weekly_by_id[False] * 100, color=LEGIT, lw=2, label="without identity"
    )
    axes[1].set_ylabel("fraud rate (%)")
    axes[1].set_xlabel("week")
    axes[1].set_title("Fraud rate by identity presence")
    axes[1].legend(frameon=False)
    save(fig, "04_identity_coverage")


FAMILY_NAMES = {
    "card": "card",
    "addr": "address",
    "dist": "distance",
    "C": "counts",
    "D": "time deltas",
    "M": "match",
    "V": "vesta",
    "id_": "identity",
}


def family_of(col: str) -> str:
    """The competition host's feature families (docs/eda.md, 'Feature families')."""
    for prefix, fam in FAMILY_NAMES.items():
        if col.startswith(prefix) and col[len(prefix) :].isdigit():
            return fam
    if col.endswith("emaildomain"):
        return "email"
    if col in ("DeviceType", "DeviceInfo"):
        return "device"
    if col == "TransactionAmt":
        return "transaction"
    if col == schema.TIME_COL:
        return "time"
    if col == "ProductCD":
        return "product"
    return col


def class_balance(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    counts = df[schema.TARGET_COL].value_counts().sort_index()
    stats["class_counts"] = {int(k): int(v) for k, v in counts.items()}
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["legit (0)", "fraud (1)"], counts.values, color=[LEGIT, FRAUD], width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n{v / counts.sum():.2%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("transactions")
    ax.set_title("Class balance")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, "00_class_balance")


BUCKETS = (
    (10, "≤10"),
    (100, "≤100"),
    (1_000, "≤1,000"),
    (100_000, "≤100,000"),
    (10**12, ">100,000"),
)
DERIVED = {"day", "week", "hour", "dow"}


def cardinality(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    """Unique-value counts for every column, bucketed by order of magnitude."""
    skip = {schema.ID_COL, schema.TARGET_COL, *DERIVED}
    rows = []
    for col in df.columns:
        if col in skip:
            continue
        n = int(df[col].nunique(dropna=True))
        bucket = (
            "≤10"
            if n <= 10
            else "≤100"
            if n <= 100
            else "≤1,000"
            if n <= 1_000
            else "≤100,000"
            if n <= 100_000
            else ">100,000"
        )
        rows.append(
            {
                "column": col,
                "family": family_of(col),
                "dtype": "string" if pd.api.types.is_string_dtype(df[col].dtype) else "numeric",
                "n_unique": n,
                "bucket": bucket,
                "null_frac": float(df[col].isna().mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("n_unique", ascending=False)
    table.to_csv(OUT / "categorical_cardinality.csv", index=False)
    stats["cardinality_buckets"] = {
        b: int((table["bucket"] == b).sum())
        for b in ("≤10", "≤100", "≤1,000", "≤100,000", ">100,000")
    }
    strings = table[table["dtype"] == "string"]
    stats["string_columns_by_cardinality"] = {
        r["column"]: int(r["n_unique"]) for _, r in strings.iterrows()
    }


def missingness_by_target(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    """Null rate per column for fraud vs legitimate rows: missingness as a signal."""
    y = df[schema.TARGET_COL] == 1
    cols = [
        c
        for c in df.columns
        if c not in (schema.ID_COL, schema.TARGET_COL, "day", "week", "hour", "dow")
    ]
    table = pd.DataFrame(
        {
            "column": cols,
            "family": [family_of(c) for c in cols],
            "null_frac": df[cols].isna().mean().to_numpy(),
            "null_frac_fraud": df.loc[y, cols].isna().mean().to_numpy(),
            "null_frac_legit": df.loc[~y, cols].isna().mean().to_numpy(),
        }
    )
    table["diff_fraud_minus_legit"] = table["null_frac_fraud"] - table["null_frac_legit"]
    table.sort_values("null_frac", ascending=False).to_csv(OUT / "missingness.csv", index=False)
    stats["missingness_by_target_top_diff"] = {
        r["column"]: {
            "fraud": round(float(r["null_frac_fraud"]), 4),
            "legit": round(float(r["null_frac_legit"]), 4),
        }
        for _, r in table.reindex(
            table["diff_fraud_minus_legit"].abs().sort_values(ascending=False).index
        )
        .head(15)
        .iterrows()
    }
    fam = (
        table.groupby("family")[["null_frac_fraud", "null_frac_legit"]]
        .mean()
        .sort_values("null_frac_legit")
    )
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(fam))
    ax.bar(x - 0.2, fam["null_frac_legit"] * 100, width=0.4, color=LEGIT, label="legit")
    ax.bar(x + 0.2, fam["null_frac_fraud"] * 100, width=0.4, color=FRAUD, label="fraud")
    ax.set_xticks(x, fam.index, rotation=30, ha="right")
    ax.set_ylabel("mean null (%) across the family's columns")
    ax.set_title("Missingness by family and label")
    ax.legend(frameon=False)
    save(fig, "08_missingness_by_target")


def missingness(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    null_frac = df.isna().mean()
    fam = null_frac.groupby(null_frac.index.map(family_of))
    stats["null_fraction_by_family"] = {
        str(k): {
            "min": float(v.min()),
            "median": float(v.median()),
            "max": float(v.max()),
            "n_cols": int(len(v)),
        }
        for k, v in fam
    }
    stats["null_fraction_top"] = {
        str(k): float(v) for k, v in null_frac.sort_values(ascending=False).head(25).items()
    }
    stats["cols_fully_populated"] = int((null_frac == 0).sum())
    stats["cols_over_90pct_null"] = int((null_frac > 0.9).sum())

    # V columns come in blocks with an identical null pattern (same rows missing).
    v = df[list(schema.V_COLS)].isna()
    pattern_key = pd.util.hash_pandas_object(v.T, index=False)
    blocks: list[list[str]] = [
        list(members.index) for _, members in pattern_key.groupby(pattern_key.values)
    ]
    blocks.sort(key=lambda b: int(b[0][1:]))
    stats["v_null_pattern_blocks"] = len(blocks)
    stats["v_blocks"] = [
        {"cols": f"{b[0]}..{b[-1]}", "n": len(b), "null_frac": float(null_frac[b[0]])}
        for b in blocks
    ]

    # Null fraction in identity columns, among rows that HAVE identity.
    idn_cols = [c for c in schema.IDENTITY_COLS if c != schema.ID_COL]
    with_id = df.loc[df[schema.HAS_IDENTITY_COL], idn_cols].isna().mean()
    stats["identity_null_fraction_given_present"] = {
        str(k): float(v) for k, v in with_id.sort_values(ascending=False).items()
    }

    tx_cols = [c for c in schema.TRANSACTION_COLS if c != schema.ID_COL]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(tx_cols)), null_frac[tx_cols].values * 100, color=LEGIT, width=1.0)
    ax.set_xlabel("transaction column (file order)")
    ax.set_ylabel("null (%)")
    ax.set_title("Null fraction per transaction column")
    ticks = [tx_cols.index(c) for c in ("TransactionDT", "card1", "C1", "D1", "M1", "V1", "V339")]
    ax.set_xticks(ticks, ["DT", "card1", "C1", "D1", "M1", "V1", "V339"])
    save(fig, "05_null_fraction_transaction_columns")


def amount(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    amt = df["TransactionAmt"]
    y = df[schema.TARGET_COL]
    stats["amount"] = {
        "min": float(amt.min()),
        "median": float(amt.median()),
        "p99": float(amt.quantile(0.99)),
        "max": float(amt.max()),
        "median_fraud": float(amt[y == 1].median()),
        "median_legit": float(amt[y == 0].median()),
        "frac_with_cents": float((amt.round(2) % 1 != 0).mean()),
        "frac_with_cents_fraud": float((amt[y == 1].round(2) % 1 != 0).mean()),
        "frac_with_cents_legit": float((amt[y == 0].round(2) % 1 != 0).mean()),
    }
    dec = pd.qcut(amt, 10, labels=False, duplicates="drop")
    by_dec = df.groupby(dec)[schema.TARGET_COL].mean()
    edges = amt.quantile(np.linspace(0, 1, 11)).round(1).tolist()
    stats["fraud_rate_by_amount_decile"] = {
        f"{edges[i]}-{edges[i + 1]}": float(v) for i, v in enumerate(by_dec.values)
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    bins = np.linspace(0, 5, 60)
    axes[0].hist(
        np.log10(amt[y == 0]), bins=bins, density=True, color=LEGIT, alpha=0.8, label="legit"
    )
    axes[0].hist(
        np.log10(amt[y == 1]), bins=bins, density=True, color=FRAUD, alpha=0.7, label="fraud"
    )
    axes[0].set_xlabel("log10(TransactionAmt)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Amount distribution by label")
    axes[0].legend(frameon=False)
    axes[1].bar(range(len(by_dec)), by_dec.values * 100, color=FRAUD, width=0.8)
    axes[1].set_xlabel("amount decile")
    axes[1].set_ylabel("fraud rate (%)")
    axes[1].set_title("Fraud rate by amount decile")
    save(fig, "06_amount")


def categoricals(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "M4"]
    out: dict[str, Any] = {}
    for c in cols:
        g = df.groupby(c, dropna=False)[schema.TARGET_COL].agg(["size", "mean"])
        g = g.sort_values("size", ascending=False).head(12)
        out[c] = {
            ("<null>" if pd.isna(k) else str(k)): {
                "n": int(r["size"]),
                "fraud_rate": float(r["mean"]),
            }
            for k, r in g.iterrows()
        }
    stats["categoricals"] = out
    stats["cardinality"] = {
        c: int(df[c].nunique())
        for c in (
            "card1",
            "card2",
            "card3",
            "card5",
            "addr1",
            "addr2",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceInfo",
            "id_30",
            "id_31",
            "id_33",
        )
    }

    g = (
        df.groupby("ProductCD")[schema.TARGET_COL]
        .agg(["size", "mean"])
        .sort_values("size", ascending=False)
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].bar(g.index, g["size"], color=LEGIT, width=0.7)
    axes[0].set_title("Volume by ProductCD")
    axes[0].set_ylabel("transactions")
    axes[1].bar(g.index, g["mean"] * 100, color=FRAUD, width=0.7)
    axes[1].set_title("Fraud rate by ProductCD")
    axes[1].set_ylabel("fraud rate (%)")
    save(fig, "07_productcd")


def d_columns_and_entities(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    """The D* trap: D1 looks like 'days since first transaction on this card'.

    If so, day - D1 is constant per card and reconstructs a stable card identifier.
    We measure how many distinct values that reconstruction takes per card1.
    """
    d = df[list(schema.D_COLS)]
    stats["d_columns"] = {
        c: {
            "null_frac": float(d[c].isna().mean()),
            "min": float(d[c].min()),
            "max": float(d[c].max()),
        }
        for c in schema.D_COLS
    }
    sub = df.loc[df["D1"].notna(), ["card1", "day", "D1", schema.TARGET_COL]].copy()
    sub["card_start_day"] = sub["day"] - sub["D1"]
    per_card1 = sub.groupby("card1")["card_start_day"].nunique()
    stats["d1_reconstruction"] = {
        "rows_with_d1": int(len(sub)),
        "n_card1": int(per_card1.size),
        "median_start_days_per_card1": float(per_card1.median()),
        "n_reconstructed_entities": int(sub.groupby(["card1", "card_start_day"]).ngroups),
    }

    # Label clustering within entity: how much fraud sits in entities with repeated fraud?
    ent = sub.groupby(["card1", "card_start_day"])[schema.TARGET_COL].agg(["sum", "size"])
    fraud_in_multi = ent.loc[ent["sum"] >= 2, "sum"].sum()
    stats["fraud_clustering"] = {
        "entities": int(len(ent)),
        "entities_with_any_fraud": int((ent["sum"] > 0).sum()),
        "entities_all_fraud": int(((ent["sum"] == ent["size"]) & (ent["sum"] > 0)).sum()),
        "fraud_rows_in_entities_with_2plus_fraud": int(fraud_in_multi),
        "fraud_rows_total_with_d1": int(ent["sum"].sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_time_columns(load_train(ROOT / "data" / "raw", cache_dir=ROOT / "data" / "processed"))
    stats: dict[str, Any] = {}
    class_balance(df, stats)
    label_and_time(df, stats)
    identity_coverage(df, stats)
    missingness(df, stats)
    missingness_by_target(df, stats)
    cardinality(df, stats)
    amount(df, stats)
    categoricals(df, stats)
    d_columns_and_entities(df, stats)
    (OUT / "dataset_summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {OUT / 'dataset_summary.json'} and {len(list(OUT.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
