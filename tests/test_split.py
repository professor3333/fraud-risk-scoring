"""Split tests: windows are time-ordered, disjoint, and sized as configured (ADR 0002)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraud.data import schema
from fraud.data.load import load_train
from fraud.data.split import (
    WINDOW_ORDER,
    SplitConfig,
    Window,
    assign_window,
    check_split,
    load_split_config,
    split,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def cfg() -> SplitConfig:
    return load_split_config(CONFIGS / "split.yaml")


@pytest.fixture(scope="module")
def frame(fixture_raw_dir: Path) -> pd.DataFrame:
    return load_train(fixture_raw_dir)


def test_config_windows_are_ordered_and_cover_the_data(cfg: SplitConfig) -> None:
    assert tuple(w.name for w in cfg.windows) == WINDOW_ORDER
    assert cfg.window("train").start_day == 1
    assert cfg.window("test").end_day == 183
    assert cfg.window("train").end_day + 1 == cfg.window("validation").start_day
    assert cfg.window("validation").end_day + 1 == cfg.window("test").start_day


def test_dev_config_shares_boundaries_and_samples() -> None:
    dev = load_split_config(CONFIGS / "dev.yaml")
    full = load_split_config(CONFIGS / "split.yaml")
    assert dev.windows == full.windows
    assert dev.sample_fraction == 0.10
    assert dev.sample_seed is not None


def test_windows_are_strictly_later_and_disjoint(frame: pd.DataFrame, cfg: SplitConfig) -> None:
    parts = split(frame, cfg)
    check_split(parts, cfg, schema.ID_COL)
    tr, va, te = parts["train"], parts["validation"], parts["test"]
    assert tr[schema.TIME_COL].max() < va[schema.TIME_COL].min()
    assert va[schema.TIME_COL].max() < te[schema.TIME_COL].min()
    ids = pd.concat([tr, va, te])[schema.ID_COL]
    assert ids.is_unique
    assert len(tr) + len(va) + len(te) == len(frame)


def test_assign_window_matches_day_boundaries(frame: pd.DataFrame, cfg: SplitConfig) -> None:
    labels = assign_window(frame, cfg)
    day = frame[schema.TIME_COL] // cfg.seconds_per_day
    for w in cfg.windows:
        expected = (day >= w.start_day) & (day <= w.end_day)
        assert (labels == w.name).equals(expected), w.name


def test_rows_outside_windows_are_dropped(frame: pd.DataFrame) -> None:
    narrow = SplitConfig(
        time_col=schema.TIME_COL,
        seconds_per_day=86_400,
        windows=(Window("train", 1, 20), Window("validation", 30, 40), Window("test", 50, 60)),
    )
    parts = split(frame, narrow)
    day = frame[schema.TIME_COL] // 86_400
    assert len(parts["train"]) == int((day <= 20).sum())
    assert len(parts["validation"]) == int(((day >= 30) & (day <= 40)).sum())
    assert sum(len(p) for p in parts.values()) < len(frame)


def test_sampling_is_deterministic(frame: pd.DataFrame) -> None:
    dev = load_split_config(CONFIGS / "dev.yaml")
    a = split(frame, dev)
    b = split(frame, dev)
    for name in WINDOW_ORDER:
        pd.testing.assert_frame_equal(a[name], b[name])
        assert a[name][schema.TIME_COL].is_monotonic_increasing


def test_overlapping_windows_rejected() -> None:
    with pytest.raises(ValueError, match="starts on day"):
        SplitConfig(
            time_col="t",
            seconds_per_day=1,
            windows=(Window("train", 1, 10), Window("validation", 10, 20), Window("test", 21, 30)),
        )


def test_wrong_window_order_rejected() -> None:
    with pytest.raises(ValueError, match="in order"):
        SplitConfig(
            time_col="t",
            seconds_per_day=1,
            windows=(Window("validation", 1, 10), Window("train", 11, 20), Window("test", 21, 30)),
        )


def test_check_split_detects_time_overlap(frame: pd.DataFrame, cfg: SplitConfig) -> None:
    parts = split(frame, cfg)
    bad = dict(parts)
    bad["validation"] = pd.concat([parts["validation"], parts["train"].tail(1)])
    with pytest.raises(ValueError, match="max time"):
        check_split(bad, cfg, schema.ID_COL)


def test_split_yaml_is_the_single_source_of_boundaries() -> None:
    raw = yaml.safe_load((CONFIGS / "split.yaml").read_text())
    assert set(raw["windows"]) == set(WINDOW_ORDER)


# --- full dataset ---------------------------------------------------------------


@pytest.mark.slow
def test_full_dataset_window_sizes(full_raw_dir: Path, cfg: SplitConfig) -> None:
    df = load_train(full_raw_dir, cache_dir=full_raw_dir.parent / "processed")
    parts = split(df, cfg)
    check_split(parts, cfg, schema.ID_COL)
    assert sum(len(p) for p in parts.values()) == len(df)
    for name in WINDOW_ORDER:
        rate = parts[name][schema.TARGET_COL].mean()
        assert 0.025 <= rate <= 0.05, (name, rate)
    assert len(parts["validation"]) > 80_000
    assert len(parts["test"]) > 80_000


# --- tuning folds (ADR 0002: expanding window inside the training window) -------


def test_tuning_folds_stay_inside_training_window(cfg: SplitConfig) -> None:
    from fraud.train.tune import check_folds_inside_training_window, load_tune_config

    tune = load_tune_config(CONFIGS / "tuning" / "xgboost.yaml")
    check_folds_inside_training_window(tune.folds, cfg)
    for f in tune.folds:
        assert f.val_end_day <= cfg.window("train").end_day
        assert f.train_end_day < f.val_start_day


def test_fold_frames_are_strictly_ordered(frame: pd.DataFrame, cfg: SplitConfig) -> None:
    from fraud.train.tune import Fold, fold_frames

    train = split(frame, cfg)["train"]
    fit, score = fold_frames(train, Fold(62, 63, 92), cfg.seconds_per_day)
    assert fit[schema.TIME_COL].max() < score[schema.TIME_COL].min()
    assert not fit[schema.ID_COL].isin(score[schema.ID_COL]).any()
    with pytest.raises(ValueError, match="ordered"):
        Fold(70, 63, 92)
