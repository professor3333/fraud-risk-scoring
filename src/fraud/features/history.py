"""Entity history features from strictly earlier rows (ADR 0004).

Pure and stateless: the frame in, a copy with five ``ent_*`` columns out. Rows
are processed in ``(TransactionDT, TransactionID)`` order; a row's features
depend only on rows of the same entity that come before it in that order. No
label is read. The entity key is a grouping device and is never emitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.data import schema

SECONDS_PER_DAY = 86_400
ENTITY_KEY_COLS: tuple[str, ...] = ("card1", "addr1", "D1")
HISTORY_FEATURES: tuple[str, ...] = (
    "ent_prior_count",
    "ent_seconds_since_prev",
    "ent_prior_amt_mean",
    "ent_amt_ratio",
    "ent_prior_count_1d",
    "ent_prior_amt_std",
    "ent_prior_amt_max",
    "ent_amt_vs_max",
)

# Named entity definitions (ADR 0004; E007 used "card_start", E017 "card_addr").
ENTITY_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "card_start": ("card1", "addr1", "D1"),
    "card_addr": ("card1", "addr1"),
}


def entity_key(df: pd.DataFrame, key_cols: tuple[str, ...] = ENTITY_KEY_COLS) -> pd.Series:
    """Join the key columns as a string; ``D1`` contributes ``day - D1`` (the card's
    start day, constant over its life). NaN when any component is missing."""
    parts: list[pd.Series] = []
    for col in key_cols:
        if col == "D1":
            start_day = df[schema.TIME_COL] // SECONDS_PER_DAY - df["D1"]
            parts.append(start_day.astype("Int64").astype(str))
        else:
            parts.append(df[col].astype("Int64").astype(str))
    complete = np.logical_and.reduce([df[c].notna().to_numpy() for c in key_cols])
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key.where(complete, other=None).astype("str")


def add_entity_history(
    df: pd.DataFrame, key_cols: tuple[str, ...] = ENTITY_KEY_COLS
) -> pd.DataFrame:
    """Append the history features; the input frame is not modified."""
    for col in (schema.TIME_COL, schema.ID_COL, "TransactionAmt", *key_cols):
        if col not in df.columns:
            raise KeyError(f"add_entity_history needs column {col!r}")

    order = np.lexsort((df[schema.ID_COL].to_numpy(), df[schema.TIME_COL].to_numpy()))
    s = df.iloc[order]
    key = entity_key(s, key_cols)
    t = s[schema.TIME_COL].to_numpy(dtype=np.float64)
    amt = s["TransactionAmt"].to_numpy(dtype=np.float64)

    g = pd.Series(np.arange(len(s)), index=s.index).groupby(key.to_numpy(), dropna=True)
    prior_count = np.full(len(s), np.nan)
    since_prev = np.full(len(s), np.nan)
    prior_mean = np.full(len(s), np.nan)
    count_1d = np.full(len(s), np.nan)
    prior_std = np.full(len(s), np.nan)
    prior_max = np.full(len(s), np.nan)

    for _, positions in g:
        pos = positions.to_numpy()
        n = len(pos)
        tt = t[pos]
        aa = amt[pos]
        prior_count[pos] = np.arange(n, dtype=np.float64)
        since_prev[pos[1:]] = np.diff(tt)
        cum = np.cumsum(aa)
        k = np.arange(1, n, dtype=np.float64)
        prior_mean[pos[1:]] = cum[:-1] / k
        # expanding population std of the earlier amounts (0 for a single earlier row)
        cum_sq = np.cumsum(aa * aa)
        var = cum_sq[:-1] / k - (cum[:-1] / k) ** 2
        prior_std[pos[1:]] = np.sqrt(np.clip(var, 0.0, None))
        prior_max[pos[1:]] = np.maximum.accumulate(aa)[:-1]
        # earlier rows within the last day: searchsorted on the sorted per-entity times
        left = np.searchsorted(tt, tt - SECONDS_PER_DAY, side="right")
        count_1d[pos] = np.arange(n) - left

    ratio = np.where(np.isnan(prior_mean) | (prior_mean == 0), np.nan, amt / prior_mean)
    vs_max = np.where(np.isnan(prior_max) | (prior_max == 0), np.nan, amt / prior_max)
    feats = pd.DataFrame(
        {
            "ent_prior_count": prior_count,
            "ent_seconds_since_prev": since_prev,
            "ent_prior_amt_mean": prior_mean,
            "ent_amt_ratio": ratio,
            "ent_prior_count_1d": count_1d,
            "ent_prior_amt_std": prior_std,
            "ent_prior_amt_max": prior_max,
            "ent_amt_vs_max": vs_max,
        },
        index=s.index,
    )
    return pd.concat([df, feats.reindex(df.index)], axis=1)
