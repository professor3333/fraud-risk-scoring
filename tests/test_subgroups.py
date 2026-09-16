"""Subgroup robustness: groupings are total, small levels pool, metrics respect the floor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud.data import schema
from fraud.data.load import load_train
from fraud.evaluate.subgroups import (
    MISSING,
    amount_band,
    email_family,
    findings,
    grouping_columns,
    subgroup_table,
    ten_day_block,
)


def test_group_definitions() -> None:
    fam = email_family(pd.Series(["gmail.com", "Hotmail.com", "anonymous.com", "att.net", None]))
    assert fam.tolist() == ["gmail", "microsoft", "anonymous", "other", MISSING]
    band = amount_band(pd.Series([0.0, 24.99, 25.0, 999.0, 1000.0, 5e4]))
    assert band.tolist() == ["0-25", "0-25", "25-50", "250-1000", ">1000", ">1000"]
    blocks = ten_day_block(pd.Series([123, 132, 133, 152]) * 86_400, first_day=123)
    assert blocks.tolist() == ["123-132", "123-132", "133-142", "143-152"]


def test_subgroup_table_on_fixture(fixture_raw_dir: Path) -> None:
    df = load_train(fixture_raw_dir)
    groups = grouping_columns(df)
    for name, col in groups.items():
        assert col.notna().all(), name  # every row belongs to exactly one level
    rng = np.random.default_rng(0)
    p = np.clip(rng.random(len(df)) * 0.5 + 0.5 * df[schema.TARGET_COL].to_numpy(), 0, 1)
    table = subgroup_table(df, p, block_t=0.8, review_t=0.3, min_positives=5, min_rows=40)
    assert set(table["grouping"]) == set(groups)
    for name in groups:
        part = table.loc[table["grouping"] == name]
        levels = part.loc[part["level"] != "all"]
        assert levels["n"].sum() == len(df)  # levels partition the frame
        assert levels["positives"].sum() == df[schema.TARGET_COL].sum()
        assert levels["share_of_rows"].sum() == pytest.approx(1.0)
        assert (levels.loc[levels["level"] != "<small>", "n"] >= 40).all()
    # the PR-AUC floor: levels with fewer positives than the floor report none
    few = table.loc[(table["level"] != "all") & (table["positives"] < 5)]
    assert few["pr_auc"].isna().all()
    both = (table["level"] != "all") & (table["positives"] >= 5) & (table["positives"] < table["n"])
    assert table.loc[both, "pr_auc"].between(0, 1 + 1e-9).all()  # AP can round past 1
    assert table["flag_rate"].between(0, 1).all() and table["block_rate"].between(0, 1).all()
    # findings fire on a fabricated failure and stay quiet on the global row
    broken = table.copy()
    i = broken.index[(broken["level"] != "all") & (broken["positives"] >= 30)]
    if len(i):
        broken.loc[i[0], ["recall_block_plus_review"]] = 0.1
        assert any("reaches 10%" in f for f in findings(broken))
    assert not any("=all" in f for f in findings(table))
