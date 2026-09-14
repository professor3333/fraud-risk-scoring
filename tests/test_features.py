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


# --- frequency encoding (ADR 0005) ------------------------------------------------


def test_frequency_encoder_learns_from_fit_rows_only() -> None:
    from fraud.features.encoders import FrequencyEncoder

    train = pd.DataFrame({"c": ["a", "a", "b", None], "n": [1.0, 1.0, 2.0, float("nan")]})
    enc = FrequencyEncoder().fit(train)
    assert enc.tables_["c"] == {"a": 0.5, "b": 0.25, "<missing>": 0.25}
    assert enc.tables_["n"] == {"1.0": 0.5, "2.0": 0.25, "<missing>": 0.25}

    later = pd.DataFrame({"c": ["a", "zzz", None], "n": [2.0, 99.0, float("nan")]})
    out = enc.transform(later)
    assert out.tolist() == [[0.5, 0.25], [0.0, 0.0], [0.25, 0.25]]
    # Transforming later rows changes nothing that was fitted.
    assert enc.tables_["c"] == {"a": 0.5, "b": 0.25, "<missing>": 0.25}
    assert list(enc.get_feature_names_out()) == ["freq_c", "freq_n"]


def test_frequency_encoder_int_and_float_keys_agree() -> None:
    from fraud.features.encoders import FrequencyEncoder

    enc = FrequencyEncoder().fit(pd.DataFrame({"n": [150, 150, 7]}))
    out = enc.transform(pd.DataFrame({"n": [150.0, 7.0, 8.0]}))
    assert out[:, 0].tolist() == [2 / 3, 1 / 3, 0.0]


def test_frequency_encoder_rejects_column_mismatch() -> None:
    from fraud.features.encoders import FrequencyEncoder

    enc = FrequencyEncoder().fit(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(ValueError, match="columns"):
        enc.transform(pd.DataFrame({"b": [2], "a": [1]}))
