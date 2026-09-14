"""Feature tests: derived features are pure, row-local, and available at prediction time."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud.data import schema
from fraud.features.derive import DERIVERS, Derive
from fraud.features.time import SECONDS_PER_DAY, add_time_features


def test_time_features_are_row_local_and_cyclic() -> None:
    df = pd.DataFrame(
        {
            schema.TIME_COL: [
                86_400,  # day 1, hour 0
                86_400 + 7 * 3600 + 59,  # day 1, hour 7
                86_400 * 8 + 23 * 3600,  # day 8 -> same weekday as day 1, hour 23
            ]
        }
    )
    out = add_time_features(df)
    assert out["hour"].tolist() == [0, 7, 23]
    assert out["weekday"].tolist() == [1, 1, 1]
    assert out["hour"].between(0, 23).all()
    assert out["weekday"].between(0, 6).all()
    # Input is untouched (pure function).
    assert "hour" not in df.columns


def test_time_features_do_not_depend_on_other_rows() -> None:
    a = pd.DataFrame({schema.TIME_COL: [SECONDS_PER_DAY * 5 + 3600 * 13]})
    b = pd.DataFrame({schema.TIME_COL: [SECONDS_PER_DAY * 5 + 3600 * 13, SECONDS_PER_DAY * 100]})
    pd.testing.assert_series_equal(
        add_time_features(a).iloc[0][["hour", "weekday"]],
        add_time_features(b).iloc[0][["hour", "weekday"]],
    )


def test_derive_step_is_stateless_and_rejects_unknown_groups() -> None:
    df = pd.DataFrame({schema.TIME_COL: [90_000]})
    step = Derive(("time",)).fit(df)
    assert set(step.transform(df).columns) == {schema.TIME_COL, "hour", "weekday"}
    with pytest.raises(ValueError, match="unknown derived"):
        Derive(("nope",)).fit(df)
    assert set(DERIVERS) == {"time"}
