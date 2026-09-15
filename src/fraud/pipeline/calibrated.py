"""A fitted pipeline plus a monotone probability map, served as one object (ADR 0007, G8)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

Method = Literal["isotonic", "sigmoid"]
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return np.asarray(np.log(q / (1 - q)), dtype=float)


class CalibratedModel(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Wrap a fitted pipeline with a monotone map fitted on *held-out* scores.

    ``sigmoid`` (Platt scaling on the logit of the score) is strictly monotone,
    so every ranking metric is unchanged. ``isotonic`` is non-parametric but
    ties scores within each step, which can move ranking metrics slightly.
    ``fit_calibrator`` must receive scores the pipeline did not train on.
    """

    def __init__(self, pipeline: Pipeline, method: Method = "sigmoid") -> None:
        self.pipeline = pipeline
        self.method = method

    def fit_calibrator(self, held_out_scores: ArrayLike, y_true: ArrayLike) -> CalibratedModel:
        s = np.asarray(held_out_scores, dtype=float)
        y = np.asarray(y_true, dtype=float)
        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrator_.fit(s, y)
        elif self.method == "sigmoid":
            self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000)
            self.calibrator_.fit(_logit(s).reshape(-1, 1), y.astype(int))
        else:
            raise ValueError(f"unknown calibration method {self.method!r}")
        self.classes_ = np.array([0, 1])
        return self

    def calibrate_scores(self, raw: ArrayLike) -> np.ndarray:
        s = np.asarray(raw, dtype=float)
        if self.method == "isotonic":
            return np.asarray(self.calibrator_.predict(s), dtype=float)
        return np.asarray(
            self.calibrator_.predict_proba(_logit(s).reshape(-1, 1))[:, 1], dtype=float
        )

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(X)[:, 1], dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.calibrate_scores(self.raw_scores(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def fit(self, X: Any = None, y: Any = None) -> CalibratedModel:
        raise NotImplementedError("fit the pipeline, then call fit_calibrator on held-out scores")
