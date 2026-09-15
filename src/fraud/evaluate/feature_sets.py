"""Feature-set ladder and family cuts, built programmatically from the base spec (E018)."""

from __future__ import annotations

from dataclasses import replace

from fraud.data import schema
from fraud.features.columns import FeatureSpec
from fraud.features.history import HISTORY_FEATURES
from fraud.features.rowwise import (
    AMOUNT_FEATURES,
    EMAIL_FAMILY_FEATURES,
    EMAIL_FEATURES,
    INTERACTION_FEATURES,
    MISSINGNESS_FEATURES,
)
from fraud.features.time import TIME_FEATURES

FREQUENCY_COLUMNS: tuple[str, ...] = (
    "card1", "card2", "card3", "card5", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_30", "id_31", "id_33",
)  # fmt: skip

LADDER: tuple[str, ...] = ("F0", "F0+F1", "+F2", "+F3", "+F4", "+F5", "+F6", "+F7")


def _add(spec: FeatureSpec, step: str) -> FeatureSpec:
    if step == "F1":
        return replace(
            spec,
            derived=spec.derived + ("missingness",),
            numeric=spec.numeric + MISSINGNESS_FEATURES,
        )
    if step == "F2":
        return replace(
            spec, derived=spec.derived + ("amount",), numeric=spec.numeric + AMOUNT_FEATURES
        )
    if step == "F3":
        return replace(spec, derived=spec.derived + ("time",), numeric=spec.numeric + TIME_FEATURES)
    if step == "F4":
        return replace(
            spec,
            derived=spec.derived + ("email",),
            numeric=spec.numeric + EMAIL_FEATURES,
            categorical=spec.categorical + EMAIL_FAMILY_FEATURES,
        )
    if step == "F5":
        return replace(
            spec,
            derived=spec.derived + ("interactions",),
            frequency=spec.frequency + INTERACTION_FEATURES,
        )
    if step == "F6":
        return replace(spec, frequency=FREQUENCY_COLUMNS + spec.frequency)
    if step == "F7":
        return replace(
            spec, history=True, history_entity="card_addr", numeric=spec.numeric + HISTORY_FEATURES
        )
    raise ValueError(step)


def ladder(base: FeatureSpec) -> dict[str, FeatureSpec]:
    """Cumulative sets F0, F0+F1, …, +F7, each built on the previous one."""
    out: dict[str, FeatureSpec] = {"F0": replace(base, name="F0")}
    spec = base
    for label, step in zip(LADDER[1:], ("F1", "F2", "F3", "F4", "F5", "F6", "F7"), strict=True):
        spec = _add(spec, step)
        out[label] = replace(spec, name=label)
    return out


# --- family cuts on the shipped feature set ------------------------------------------

_IDENTITY = frozenset(c for c in schema.IDENTITY_COLS if c != schema.ID_COL) | {
    schema.HAS_IDENTITY_COL
}
_V = frozenset(schema.V_COLS)
_VESTA = _V | frozenset(schema.C_COLS) | frozenset(schema.D_COLS) | frozenset(schema.M_COLS)


def _without(spec: FeatureSpec, cols: frozenset[str], name: str) -> FeatureSpec:
    return replace(
        spec,
        name=name,
        numeric=tuple(c for c in spec.numeric if c not in cols),
        categorical=tuple(c for c in spec.categorical if c not in cols),
        frequency=tuple(c for c in spec.frequency if c not in cols),
    )


def family_cuts(shipped: FeatureSpec, raw: FeatureSpec) -> dict[str, FeatureSpec]:
    """Family cuts as a 2 x 2 of Vesta's V block x my features, plus the no-Vesta floor.

    ``raw`` is the provider's columns only (F0); ``shipped`` adds frequency tables
    and composite keys.
    """
    return {
        "raw transaction only (no identity, no V)": _without(raw, _IDENTITY | _V, "raw_tx_only"),
        "raw transaction + identity (no V)": _without(raw, _V, "raw_noV"),
        "raw everything incl. V (F0)": replace(raw, name="raw_all"),
        "shipped, transaction only (no identity, no V)": _without(
            shipped, _IDENTITY | _V, "shipped_tx_only"
        ),
        "shipped except V": _without(shipped, _V, "shipped_noV"),
        "shipped (everything)": replace(shipped, name="shipped_all"),
        "shipped except Vesta-engineered (C, D, M, V)": _without(
            shipped, _VESTA, "shipped_noVesta"
        ),
    }
