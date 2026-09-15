"""Feature tests: derived features are pure, row-local, and available at prediction time."""

from __future__ import annotations

import numpy as np
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
    assert set(DERIVERS) == {"time", "missingness", "amount", "email", "interactions"}


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


# --- entity history (ADR 0004) --------------------------------------------------


def _history_frame() -> pd.DataFrame:
    """Two entities interleaved in time, one row with a missing key."""
    return pd.DataFrame(
        {
            schema.ID_COL: [1, 2, 3, 4, 5, 6, 7],
            schema.TIME_COL: [
                100_000,  # A first
                100_500,  # B first
                101_000,  # A second (1000 s later)
                150_000,  # A third (49 000 s later)
                190_000,  # B second
                190_100,  # A fourth: 40 000 s after third, > 1 day after first two
                200_000,  # missing D1 -> no entity
            ],
            "TransactionAmt": [10.0, 5.0, 30.0, 20.0, 5.0, 100.0, 7.0],
            "card1": [1, 2, 1, 1, 2, 1, 1],
            "addr1": [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0],
            # day - D1 constant per entity: days are 1,1,1,1,2,2,2
            "D1": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, float("nan")],
        }
    )


def test_entity_history_uses_only_strictly_earlier_rows() -> None:
    from fraud.features.history import HISTORY_FEATURES, add_entity_history

    out = add_entity_history(_history_frame())
    a = out[out["card1"] == 1].iloc[:4]  # rows 1,3,4,6 in time order
    assert a["ent_prior_count"].tolist() == [0, 1, 2, 3]
    assert a["ent_seconds_since_prev"].tolist()[1:] == [1_000, 49_000, 40_100]
    assert pd.isna(a["ent_seconds_since_prev"].iloc[0])
    assert a["ent_prior_amt_mean"].tolist()[1:] == [10.0, 20.0, 20.0]
    assert a["ent_amt_ratio"].tolist()[1:] == [3.0, 1.0, 5.0]
    # within the previous day: row 3 sees row 1; row 4 sees rows 1,3 (50 000 s < 86 400);
    # row 6 sees only row 4 (rows 1,3 are > 1 day back)
    assert a["ent_prior_count_1d"].tolist() == [0, 1, 2, 1]
    b = out[out["card1"] == 2]
    assert b["ent_prior_count"].tolist() == [0, 1]
    assert out.loc[out[schema.ID_COL] == 7, list(HISTORY_FEATURES)].isna().all(axis=None)
    # Input untouched; output has the same row order and index.
    assert "ent_prior_count" not in _history_frame().columns
    assert out.index.equals(_history_frame().index)


def test_entity_history_is_unaffected_by_later_rows() -> None:
    """Altering or appending later rows must not change earlier rows' features (G3)."""
    from fraud.features.history import HISTORY_FEATURES, add_entity_history

    base = _history_frame()
    before = add_entity_history(base)
    later = base.copy()
    later.loc[later[schema.TIME_COL] >= 150_000, "TransactionAmt"] *= 100
    extra = pd.DataFrame(
        {
            schema.ID_COL: [8, 9],
            schema.TIME_COL: [250_000, 260_000],
            "TransactionAmt": [1.0, 2.0],
            "card1": [1, 2],
            "addr1": [300.0, 300.0],
            "D1": [2.0, 2.0],
        }
    )
    after = add_entity_history(pd.concat([later, extra], ignore_index=True))
    early = base[schema.TIME_COL] < 150_000
    pd.testing.assert_frame_equal(
        before.loc[early, list(HISTORY_FEATURES)].reset_index(drop=True),
        after.loc[early.to_numpy().tolist() + [False, False], list(HISTORY_FEATURES)].reset_index(
            drop=True
        ),
    )


def test_entity_history_is_order_independent() -> None:
    from fraud.features.history import HISTORY_FEATURES, add_entity_history

    base = _history_frame()
    shuffled = base.sample(frac=1.0, random_state=3)
    a = add_entity_history(base).set_index(schema.ID_COL)[list(HISTORY_FEATURES)]
    b = add_entity_history(shuffled).set_index(schema.ID_COL)[list(HISTORY_FEATURES)]
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_entity_key_is_not_emitted_as_a_feature() -> None:
    from fraud.features.history import HISTORY_FEATURES, add_entity_history

    out = add_entity_history(_history_frame())
    new_cols = set(out.columns) - set(_history_frame().columns)
    assert new_cols == set(HISTORY_FEATURES)


# --- row-local feature sets F1, F2, F4, F5 --------------------------------------------


def _one_row(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        c: float("nan") for c in schema.TRANSACTION_COLS + schema.IDENTITY_COLS[1:]
    }
    row.update(
        {
            schema.ID_COL: 1,
            schema.TARGET_COL: 0,
            schema.TIME_COL: 100_000,
            "TransactionAmt": 125.99,
            "ProductCD": "W",
            "card1": 1000,
            "addr1": 300.0,
            "card4": "visa",
            "card6": "debit",
        }
    )
    row.update(overrides)
    return pd.DataFrame([row])


def test_missingness_counts() -> None:
    from fraud.features.rowwise import add_missingness_counts

    out = add_missingness_counts(_one_row())
    # 392 counted transaction columns (id and label excluded), 7 filled; 40 identity columns
    assert out["n_missing_transaction"].item() == 392 - 7
    assert out["n_missing_identity"].item() == 40
    assert out["n_missing_total"].item() == 392 - 7 + 40


def test_amount_features_decompose_the_amount() -> None:
    from fraud.features.rowwise import add_amount_features

    out = add_amount_features(pd.DataFrame({"TransactionAmt": [125.99, 50.0, 12.5, 107.0, 0.251]}))
    assert out["amt_integer_part"].tolist() == [125.0, 50.0, 12.0, 107.0, 0.0]
    assert out["amt_fraction"].tolist() == [0.99, 0.0, 0.5, 0.0, 0.25]
    assert out["amt_is_round"].tolist() == [0, 1, 0, 0, 0]
    assert out["amt_cents_digits"].tolist() == [2, 0, 1, 0, 2]
    assert out["amt_log"].iloc[0] == pytest.approx(np.log1p(125.99))


def test_email_features_and_families() -> None:
    from fraud.features.rowwise import add_email_features

    df = pd.DataFrame(
        {
            "P_emaildomain": ["gmail.com", "hotmail.com", None, "weird.io"],
            "R_emaildomain": ["gmail.com", "outlook.com", None, None],
        }
    )
    out = add_email_features(df)
    assert out["same_email_domain"].tolist() == [1, 0, 0, 0]
    assert out["p_email_missing"].tolist() == [0, 0, 1, 0]
    assert out["r_email_missing"].tolist() == [0, 0, 1, 1]
    assert out["p_email_family"].tolist() == ["gmail", "microsoft", "missing", "other"]
    assert out["r_email_family"].tolist() == ["gmail", "microsoft", "missing", "missing"]


def test_interaction_keys_are_missing_when_any_part_is() -> None:
    from fraud.features.rowwise import add_interaction_keys

    out = add_interaction_keys(_one_row(P_emaildomain="gmail.com"))
    assert out["card1_addr1"].item() == "1000|300"
    assert out["card1_addr1_pemail"].item() == "1000|300|gmail.com"
    assert out["card1_card4"].item() == "1000|visa"
    assert pd.isna(add_interaction_keys(_one_row(addr1=float("nan")))["card1_addr1"].item())
    # non-integer numbers must not break the key (integer-valued ones keep the same string)
    assert add_interaction_keys(_one_row(addr1=300.5))["card1_addr1"].item() == "1000|300.5"
    assert add_interaction_keys(_one_row(card1=1000.0))["card1_addr1"].item() == "1000|300"


def test_all_derivers_are_row_local() -> None:
    from fraud.features.derive import DERIVERS

    a = _one_row(P_emaildomain="gmail.com")
    b = pd.concat([a, _one_row(TransactionAmt=9999.0, card1=5, addr1=1.0)], ignore_index=True)
    for name, fn in DERIVERS.items():
        left = fn(a).iloc[0]
        right = fn(b).iloc[0]
        pd.testing.assert_series_equal(left, right, check_names=False, obj=name)


def test_entity_history_std_max_and_alternate_key() -> None:
    from fraud.features.history import ENTITY_DEFINITIONS, add_entity_history

    out = add_entity_history(_history_frame(), ENTITY_DEFINITIONS["card_addr"])
    a = out[out["card1"] == 1].sort_values(schema.TIME_COL)
    # card_addr key ignores D1: row 7 (D1 missing) now joins card 1's entity as its 5th row
    assert a["ent_prior_count"].tolist() == [0, 1, 2, 3, 4]
    # earlier amounts for row 4: [10, 30] -> std 10, max 30; row 6: [10, 30, 20] -> max 30
    assert a["ent_prior_amt_std"].tolist()[2] == pytest.approx(10.0)
    assert a["ent_prior_amt_max"].tolist()[1:] == [10.0, 30.0, 30.0, 100.0]
    assert a["ent_amt_vs_max"].tolist()[3] == pytest.approx(100.0 / 30.0)
    assert pd.isna(a["ent_prior_amt_std"].iloc[0])
