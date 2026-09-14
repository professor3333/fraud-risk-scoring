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
)


def entity_key(df: pd.DataFrame) -> pd.Series:
    """``card1 | addr1 | day - D1`` as a string; NaN when any component is missing."""
    day = df[schema.TIME_COL] // SECONDS_PER_DAY
    start_day = day - df["D1"]
    complete = df["card1"].notna() & df["addr1"].notna() & df["D1"].notna()
    key = (
        df["card1"].astype("Int64").astype(str)
        + "|"
        + df["addr1"].astype("Int64").astype(str)
        + "|"
        + start_day.astype("Int64").astype(str)
    )
    return key.where(complete, other=None).astype("str")


def add_entity_history(df: pd.DataFrame) -> pd.DataFrame:
    """Append the five history features; the input frame is not modified."""
    for col in (schema.TIME_COL, schema.ID_COL, "TransactionAmt", *ENTITY_KEY_COLS):
        if col not in df.columns:
            raise KeyError(f"add_entity_history needs column {col!r}")

    order = np.lexsort((df[schema.ID_COL].to_numpy(), df[schema.TIME_COL].to_numpy()))
    s = df.iloc[order]
    key = entity_key(s)
    t = s[schema.TIME_COL].to_numpy(dtype=np.float64)
    amt = s["TransactionAmt"].to_numpy(dtype=np.float64)

    g = pd.Series(np.arange(len(s)), index=s.index).groupby(key.to_numpy(), dropna=True)
    prior_count = np.full(len(s), np.nan)
    since_prev = np.full(len(s), np.nan)
    prior_mean = np.full(len(s), np.nan)
    count_1d = np.full(len(s), np.nan)

    for _, positions in g:
        pos = positions.to_numpy()
        n = len(pos)
        tt = t[pos]
        aa = amt[pos]
        prior_count[pos] = np.arange(n, dtype=np.float64)
        since_prev[pos[1:]] = np.diff(tt)
        cum = np.cumsum(aa)
        prior_mean[pos[1:]] = cum[:-1] / np.arange(1, n, dtype=np.float64)
        # earlier rows within the last day: searchsorted on the sorted per-entity times
        left = np.searchsorted(tt, tt - SECONDS_PER_DAY, side="right")
        count_1d[pos] = np.arange(n) - left

    ratio = np.where(np.isnan(prior_mean) | (prior_mean == 0), np.nan, amt / prior_mean)
    feats = pd.DataFrame(
        {
            "ent_prior_count": prior_count,
            "ent_seconds_since_prev": since_prev,
            "ent_prior_amt_mean": prior_mean,
            "ent_amt_ratio": ratio,
            "ent_prior_count_1d": count_1d,
        },
        index=s.index,
    )
    return pd.concat([df, feats.reindex(df.index)], axis=1)
