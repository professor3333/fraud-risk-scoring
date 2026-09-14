"""A fitted pipeline plus a monotone probability map, served as one object (ADR 0007, G8)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline


class CalibratedModel(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Wrap a fitted pipeline with an isotonic map fitted on *held-out* scores.

    The map is monotone, so rankings — and every ranking metric — are unchanged;
    only the probability scale moves. ``fit_calibrator`` must receive scores the
    pipeline did not train on (out-of-fold predictions from the training window).
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def fit_calibrator(self, held_out_scores: ArrayLike, y_true: ArrayLike) -> CalibratedModel:
        self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator_.fit(
            np.asarray(held_out_scores, dtype=float), np.asarray(y_true, dtype=float)
        )
        self.classes_ = np.array([0, 1])
        return self

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(X)[:, 1], dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.calibrator_.predict(self.raw_scores(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def fit(self, X: Any = None, y: Any = None) -> CalibratedModel:
        raise NotImplementedError(
            "fit the pipeline first, then call fit_calibrator on held-out scores"
        )
