"""Pipeline assembly: preprocessing + model, as one sklearn ``Pipeline`` (G7, G8).

The pipeline takes the raw joined frame and selects its own columns, so training
and serving feed it the same thing. Every fitted statistic (medians, scaler
moments, one-hot vocabularies) is learned by ``fit`` on whatever it is given —
the caller is responsible for giving it the training window only.

Two preprocessing variants exist because the models need different inputs:

- ``linear``: median imputation + missing indicators + standard scaling, for
  logistic regression, which cannot take NaN and is scale-sensitive.
- ``tree``: numeric columns cast to float with NaN kept, for gradient-boosted
  trees, which route missing values natively and are scale-invariant.

Both one-hot encode the same low-cardinality categoricals with ``<missing>`` as
a level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud.features.columns import FeatureSpec
from fraud.features.derive import Derive
from fraud.features.encoders import FrequencyEncoder

MISSING_LEVEL = "<missing>"
PREPROCESSING_FOR_MODEL: dict[str, str] = {
    "constant": "linear",
    "logreg": "linear",
    "xgboost": "tree",
}


class MissingAsCategory(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Turn string columns into object arrays where NaN is an explicit level.

    Missingness is informative here (docs/eda.md §4), so it becomes a category
    rather than being dropped or imputed.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> MissingAsCategory:
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.astype(object).where(X.notna(), MISSING_LEVEL).to_numpy(dtype=object)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(input_features, dtype=object)


class ToFloat(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Cast numeric / boolean columns to float64 so downstream steps see one dtype."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> ToFloat:
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy(dtype=np.float64, na_value=np.nan)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(input_features, dtype=object)


def build_preprocessor(spec: FeatureSpec, kind: str) -> ColumnTransformer:
    if kind == "linear":
        numeric = Pipeline(
            [
                ("to_float", ToFloat()),
                (
                    "impute",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scale", StandardScaler()),
            ]
        )
    elif kind == "tree":
        numeric = Pipeline([("to_float", ToFloat())])
    else:
        raise ValueError(f"unknown preprocessing kind {kind!r}")
    categorical = Pipeline(
        [
            ("missing", MissingAsCategory()),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = [
        ("num", numeric, list(spec.numeric)),
        ("cat", categorical, list(spec.categorical)),
    ]
    if spec.frequency:
        transformers.append(("freq", FrequencyEncoder(), list(spec.frequency)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def build_model(model_cfg: dict[str, Any], seed: int) -> BaseEstimator:
    kind = model_cfg["type"]
    params: dict[str, Any] = dict(model_cfg.get("params", {}))
    if kind == "constant":
        return DummyClassifier(strategy="prior")
    if kind == "logreg":
        return LogisticRegression(random_state=seed, **params)
    if kind == "xgboost":
        return XGBClassifier(random_state=seed, tree_method="hist", **params)
    raise ValueError(f"unknown model type {kind!r}")


def build_pipeline(spec: FeatureSpec, model_cfg: dict[str, Any], seed: int) -> Pipeline:
    kind = PREPROCESSING_FOR_MODEL[model_cfg["type"]]
    return Pipeline(
        [
            ("derive", Derive(spec.derived)),
            ("features", build_preprocessor(spec, kind)),
            ("model", build_model(model_cfg, seed)),
        ]
    )
