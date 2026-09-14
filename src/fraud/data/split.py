"""Temporal train / validation / test windows (ADR 0002).

The only module that interprets ``TransactionDT`` as time for splitting. Window
boundaries come from a YAML config; nothing here is fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

WINDOW_ORDER: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class Window:
    name: str
    start_day: int
    end_day: int  # inclusive

    def __post_init__(self) -> None:
        if self.start_day > self.end_day:
            raise ValueError(f"{self.name}: start_day {self.start_day} > end_day {self.end_day}")


@dataclass(frozen=True)
class SplitConfig:
    time_col: str
    seconds_per_day: int
    windows: tuple[Window, ...]
    sample_fraction: float | None = None
    sample_seed: int | None = None

    def __post_init__(self) -> None:
        names = tuple(w.name for w in self.windows)
        if names != WINDOW_ORDER:
            raise ValueError(f"windows must be {WINDOW_ORDER} in order, got {names}")
        for earlier, later in zip(self.windows, self.windows[1:], strict=False):
            if earlier.end_day >= later.start_day:
                raise ValueError(
                    f"{earlier.name} ends on day {earlier.end_day}, "
                    f"{later.name} starts on day {later.start_day}"
                )
        if (self.sample_fraction is None) != (self.sample_seed is None):
            raise ValueError("sample.fraction and sample.seed must be given together")
        if self.sample_fraction is not None and not 0 < self.sample_fraction <= 1:
            raise ValueError(f"sample.fraction must be in (0, 1], got {self.sample_fraction}")

    def window(self, name: str) -> Window:
        return next(w for w in self.windows if w.name == name)

    def start_seconds(self, name: str) -> int:
        return self.window(name).start_day * self.seconds_per_day

    def end_seconds_exclusive(self, name: str) -> int:
        return (self.window(name).end_day + 1) * self.seconds_per_day


def load_split_config(path: Path) -> SplitConfig:
    raw = yaml.safe_load(path.read_text())
    windows = tuple(
        Window(
            name=n,
            start_day=int(raw["windows"][n]["start_day"]),
            end_day=int(raw["windows"][n]["end_day"]),
        )
        for n in raw["windows"]
    )
    sample = raw.get("sample") or {}
    return SplitConfig(
        time_col=str(raw["time_col"]),
        seconds_per_day=int(raw["seconds_per_day"]),
        windows=windows,
        sample_fraction=None if "fraction" not in sample else float(sample["fraction"]),
        sample_seed=None if "seed" not in sample else int(sample["seed"]),
    )


def assign_window(df: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Return a string Series naming each row's window; rows outside every window get ``""``."""
    t = df[cfg.time_col]
    out = pd.Series("", index=df.index, dtype="str")
    for w in cfg.windows:
        mask = (t >= cfg.start_seconds(w.name)) & (t < cfg.end_seconds_exclusive(w.name))
        out[mask] = w.name
    return out


def split(df: pd.DataFrame, cfg: SplitConfig) -> dict[str, pd.DataFrame]:
    """Split ``df`` into the configured windows, preserving row order within each.

    Rows outside all windows are dropped. If the config carries a sample, each
    window is independently subsampled with the configured seed.
    """
    labels = assign_window(df, cfg)
    out: dict[str, pd.DataFrame] = {}
    for w in cfg.windows:
        part = df.loc[labels == w.name]
        if cfg.sample_fraction is not None:
            part = part.sample(frac=cfg.sample_fraction, random_state=cfg.sample_seed).sort_index()
        out[w.name] = part
    return out


def check_split(parts: dict[str, pd.DataFrame], cfg: SplitConfig, id_col: str) -> None:
    """Raise if the windows are not strictly time-ordered and disjoint."""
    for earlier, later in zip(WINDOW_ORDER, WINDOW_ORDER[1:], strict=False):
        a, b = parts[earlier], parts[later]
        if a.empty or b.empty:
            raise ValueError(f"window {earlier if a.empty else later} is empty")
        if a[cfg.time_col].max() >= b[cfg.time_col].min():
            raise ValueError(f"{earlier} max time >= {later} min time")
        if a[id_col].isin(b[id_col]).any():
            raise ValueError(f"{earlier} and {later} share {id_col}s")
