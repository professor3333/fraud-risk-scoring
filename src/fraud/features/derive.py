"""Registry of derived-feature functions and the pipeline step that applies them.

Each function is pure (frame in, frame out) and uses only the row's own fields,
so it is safe at serving time. Time-aware features that need earlier rows live
in their own module with their own contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fraud.features.time import add_time_features

DERIVERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "time": add_time_features,
}


class Derive(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Apply the named derivers in order. Stateless: ``fit`` learns nothing."""

    def __init__(self, names: tuple[str, ...] = ()) -> None:
        self.names = names

    def fit(self, X: pd.DataFrame, y: Any = None) -> Derive:
        unknown = set(self.names) - set(DERIVERS)
        if unknown:
            raise ValueError(f"unknown derived feature groups {sorted(unknown)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X
        for name in self.names:
            out = DERIVERS[name](out)
        return out
