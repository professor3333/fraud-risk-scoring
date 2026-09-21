"""Drift arithmetic: population stability index and binned distributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PSI_WARN = 0.10
PSI_ALERT = 0.20
EPS = 1e-6


def psi(reference: dict[str, float], current: dict[str, float]) -> float:
    """Population stability index between two share dictionaries (same keys expected)."""
    # Sorted, not set order: the sum is over floats, and Python randomises string hashing
    # per process, so an unordered sum makes the same window differ in the last bit between
    # the service and an offline run. The scheduled monitor compares the two (ADR 0011).
    keys = sorted(set(reference) | set(current))
    total = 0.0
    for k in keys:
        r = max(float(reference.get(k, 0.0)), EPS)
        c = max(float(current.get(k, 0.0)), EPS)
        total += (c - r) * np.log(c / r)
    return float(total)


def level(value: float) -> str:
    return "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "ok"


def categorical_shares(
    values: pd.Series, top: int = 24, keys: set[str] | None = None
) -> dict[str, float]:
    """Share per level (missing counted as '<missing>').

    With ``keys`` (the reference's levels) every other level folds into '<other>',
    so a window is always compared on the reference's own partition; without it,
    the ``top`` most frequent levels are kept and the rest folded.
    """
    s = values.astype("object").where(values.notna(), "<missing>").astype(str)
    counts = s.value_counts(normalize=True)
    if keys is not None:
        known = {k for k in keys if k != "<other>"}
        shares = {k: float(counts.get(k, 0.0)) for k in known}
        rest = float(counts[~counts.index.isin(known)].sum())
    else:
        keep = counts.head(top)
        shares = {str(k): float(v) for k, v in keep.items()}
        rest = float(counts.iloc[top:].sum()) if len(counts) > top else 0.0
    if rest > 0:
        shares["<other>"] = rest
    return shares


def binned_shares(values: pd.Series, edges: list[float]) -> dict[str, float]:
    """Share per bin for a numeric series, using fixed edges (reference-defined)."""
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return {}
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1) / len(v)
    return {f"{edges[i]:g}-{edges[i + 1]:g}": float(c) for i, c in enumerate(counts)}


def quantile_edges(values: pd.Series, n_bins: int = 10) -> list[float]:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    q = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    q[0], q[-1] = -np.inf, np.inf
    return [float(x) for x in q]


def fixed_width_edges(values: pd.Series, n_bins: int = 8) -> list[float]:
    """Equal-width bins over the 1st–99th percentile range, open at both ends.

    For discrete counts (e.g. number of missing fields) quantile edges collapse onto
    a few repeated values and turn tiny shuffles into large PSI; equal widths do not.
    """
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    lo, hi = float(np.quantile(v, 0.01)), float(np.quantile(v, 0.99))
    if hi <= lo:
        return [-np.inf, lo, np.inf]
    inner = np.linspace(lo, hi, n_bins + 1)[1:-1]
    return [-np.inf, *[float(x) for x in inner], np.inf]


def numeric_summary(values: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return {"n": 0}
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "p50": float(v.quantile(0.5)),
        "p90": float(v.quantile(0.9)),
        "p99": float(v.quantile(0.99)),
    }
