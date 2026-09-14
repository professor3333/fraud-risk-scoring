"""Cyclical time features derived from ``TransactionDT`` (leakage audit: hour / weekday).

Pure: takes a frame, returns a copy with the derived columns. The reference
origin of ``TransactionDT`` is undisclosed, so ``hour`` and ``weekday`` are
offsets in the provider's clock, not wall-clock values; EDA shows they carry a
stable daily and weekly cycle. The raw timestamp itself is never a feature.
"""

from __future__ import annotations

import pandas as pd

from fraud.data import schema

SECONDS_PER_DAY = 86_400
TIME_FEATURES: tuple[str, ...] = ("hour", "weekday")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df[schema.TIME_COL]
    day = t // SECONDS_PER_DAY
    return df.assign(
        hour=((t % SECONDS_PER_DAY) // 3600).astype("int64"),
        weekday=(day % 7).astype("int64"),
    )
