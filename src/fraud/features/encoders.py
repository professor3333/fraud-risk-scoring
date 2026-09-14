"""Fitted categorical encoders. Everything here learns from ``fit`` input only (G2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MISSING_KEY = "<missing>"


class FrequencyEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Replace each value with its share of the rows seen at ``fit`` time (ADR 0005).

    Missing values are a level of their own. Values not seen at fit time map to
    0.0 — the honest statement that they were never observed in training.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> FrequencyEncoder:
        self.columns_ = list(X.columns)
        self.n_fit_ = int(len(X))
        self.tables_: dict[str, dict[str, float]] = {}
        for col in self.columns_:
            keys = _as_keys(X[col])
            counts = keys.value_counts(dropna=False)
            self.tables_[col] = {str(k): float(v) / self.n_fit_ for k, v in counts.items()}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if list(X.columns) != self.columns_:
            raise ValueError(f"columns {list(X.columns)} != fitted {self.columns_}")
        out = np.empty((len(X), len(self.columns_)), dtype=np.float64)
        for j, col in enumerate(self.columns_):
            table = self.tables_[col]
            out[:, j] = _as_keys(X[col]).map(table).fillna(0.0).to_numpy(dtype=np.float64)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([f"freq_{c}" for c in self.columns_], dtype=object)


def _as_keys(s: pd.Series) -> pd.Series:
    """Stable string keys: floats like 150.0 and ints like 150 must agree; NaN is a level."""
    if pd.api.types.is_numeric_dtype(s.dtype):
        keys = s.map(lambda v: MISSING_KEY if pd.isna(v) else repr(float(v)))
    else:
        keys = s.astype(object).where(s.notna(), MISSING_KEY).astype(str)
    return pd.Series(keys, index=s.index, dtype=object)
