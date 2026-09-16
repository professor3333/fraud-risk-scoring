"""Subgroup robustness: the global score, broken down by who the transaction is.

No demographic attributes exist in this data, so this is not a fairness
analysis. It asks a narrower question the model card left open: does one
validation PR-AUC hide groups on which the model, or the served policy, fails —
product, card network and type, identity presence, e-mail provider family,
device class, amount band, and the ten-day block inside the window.

Every number is computed on the rows of one group with the served policy's
thresholds held fixed (the policy is global; per-group thresholds would be a
different policy). Ranking metrics are suppressed below a minimum number of
positives — a PR-AUC on twelve frauds is a coin toss, not a finding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.data import schema

SECONDS_PER_DAY = 86_400
MISSING = "<missing>"

EMAIL_FAMILIES: dict[str, tuple[str, ...]] = {
    "gmail": ("gmail.com", "gmail"),
    "yahoo": ("yahoo.com", "yahoo.com.mx", "yahoo.co.uk", "yahoo.fr", "yahoo.es", "yahoo.de",
              "yahoo.co.jp", "ymail.com", "rocketmail.com"),
    "microsoft": ("hotmail.com", "outlook.com", "live.com", "msn.com", "hotmail.es",
                  "hotmail.fr", "hotmail.de", "hotmail.co.uk", "outlook.es", "live.com.mx",
                  "live.fr"),
    "apple": ("icloud.com", "me.com", "mac.com"),
    "aol": ("aol.com", "aim.com"),
    "anonymous": ("anonymous.com",),
}  # fmt: skip
_FAMILY_OF = {domain: family for family, domains in EMAIL_FAMILIES.items() for domain in domains}

AMOUNT_EDGES = (0.0, 25.0, 50.0, 100.0, 250.0, 1000.0, np.inf)


def email_family(domain: pd.Series) -> pd.Series:
    """Provider family of an e-mail domain: the big consumer providers, anonymous,
    missing, and 'other' (ISPs, corporate, regional)."""
    lowered = (
        domain.astype("object")
        .where(domain.notna(), None)
        .map(lambda d: None if d is None else str(d).lower())
    )
    return pd.Series(
        np.where(lowered.isna(), MISSING, lowered.map(lambda d: _FAMILY_OF.get(d, "other"))),
        index=domain.index,
        dtype="object",
    )


def amount_band(amount: pd.Series) -> pd.Series:
    labels = [
        f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        for lo, hi in zip(AMOUNT_EDGES[:-1], AMOUNT_EDGES[1:], strict=True)
    ]
    return pd.cut(amount, AMOUNT_EDGES, labels=labels, right=False, include_lowest=True).astype(
        "object"
    )


def ten_day_block(dt: pd.Series, first_day: int) -> pd.Series:
    day = dt // SECONDS_PER_DAY
    start = first_day + ((day - first_day) // 10) * 10
    return (start.astype(str) + "-" + (start + 9).astype(str)).astype("object")


def _fill(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING)


def grouping_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """The subgroup definitions, in the order they are reported."""
    first_day = int(frame[schema.TIME_COL].min() // SECONDS_PER_DAY)
    return {
        "product": _fill(frame["ProductCD"]),
        "card_network": _fill(frame["card4"]),
        "card_type": _fill(frame["card6"]),
        "identity": frame[schema.HAS_IDENTITY_COL].map({True: "present", False: "absent"}),
        "email_family": email_family(frame["P_emaildomain"]),
        "device_type": _fill(frame["DeviceType"]),
        "amount_band": amount_band(frame["TransactionAmt"]),
        "ten_day_block": ten_day_block(frame[schema.TIME_COL], first_day),
    }


def group_metrics(
    y: np.ndarray, p: np.ndarray, block_t: float, review_t: float, min_positives: int
) -> dict[str, float | None]:
    n, n_pos = len(y), int(y.sum())
    blocked, flagged = p >= block_t, p >= review_t
    n_blk = int(blocked.sum())
    enough = n_pos >= min_positives and n_pos < n
    return {
        "n": float(n),
        "positives": float(n_pos),
        "prevalence": n_pos / n if n else None,
        "mean_score": float(p.mean()) if n else None,
        "pr_auc": float(average_precision_score(y, p)) if enough else None,
        "block_precision": float(y[blocked].mean()) if n_blk else None,
        "recall_block": float(y[blocked].sum() / n_pos) if n_pos else None,
        "recall_block_plus_review": float(y[flagged].sum() / n_pos) if n_pos else None,
        "flag_rate": float(flagged.mean()) if n else None,
        "block_rate": float(blocked.mean()) if n else None,
    }


def subgroup_table(
    frame: pd.DataFrame,
    p: np.ndarray,
    block_t: float,
    review_t: float,
    min_positives: int = 30,
    min_rows: int = 500,
) -> pd.DataFrame:
    """One row per (grouping, level) plus an 'all' row per grouping; levels with fewer
    than ``min_rows`` rows are pooled into '<small>' so nothing is hidden and nothing
    tiny is over-read."""
    y = frame[schema.TARGET_COL].to_numpy(dtype=int)
    p = np.asarray(p, dtype=float)
    rows: list[dict[str, Any]] = []
    for name, col in grouping_columns(frame).items():
        counts = col.value_counts()
        level = col.where(col.map(counts) >= min_rows, "<small>")
        rows.append(
            {
                "grouping": name,
                "level": "all",
                **group_metrics(y, p, block_t, review_t, min_positives),
            }
        )
        for lvl in sorted(level.unique(), key=lambda v: (v == "<small>", str(v))):
            mask = (level == lvl).to_numpy()
            rows.append(
                {
                    "grouping": name,
                    "level": str(lvl),
                    **group_metrics(y[mask], p[mask], block_t, review_t, min_positives),
                }
            )
    table = pd.DataFrame(rows)
    overall = table.loc[table["level"] == "all"].set_index("grouping")
    table["delta_pr_auc"] = table["pr_auc"] - table["grouping"].map(overall["pr_auc"])
    table["share_of_rows"] = table["n"] / table["grouping"].map(overall["n"])
    table["share_of_fraud"] = table["positives"] / table["grouping"].map(overall["positives"])
    return table


def findings(
    table: pd.DataFrame, pr_auc_drop: float = 0.10, precision_floor: float = 0.70
) -> list[str]:
    """Levels that are far from the global picture: a large PR-AUC drop, block precision
    below the bar, or a recall that says the policy barely reaches the group."""
    out: list[str] = []
    for _, r in table.loc[table["level"] != "all"].iterrows():
        tag = f"{r['grouping']}={r['level']} (n {int(r['n'])}, {int(r['positives'])} fraud)"
        if pd.notna(r["delta_pr_auc"]) and r["delta_pr_auc"] <= -pr_auc_drop:
            out.append(
                f"{tag}: PR-AUC {r['pr_auc']:.3f} ({r['delta_pr_auc']:+.3f} vs the grouping)"
            )
        if (
            pd.notna(r["block_precision"])
            and r["block_precision"] < precision_floor
            and r["block_rate"] * r["n"] >= 20
        ):
            n_blocks = int(r["block_rate"] * r["n"])
            out.append(f"{tag}: block precision {r['block_precision']:.2f} on {n_blocks} blocks")
        if (
            pd.notna(r["recall_block_plus_review"])
            and r["positives"] >= 30
            and r["recall_block_plus_review"] < 0.5
        ):
            out.append(
                f"{tag}: block + review reaches {r['recall_block_plus_review']:.0%} of its fraud"
            )
    return out
