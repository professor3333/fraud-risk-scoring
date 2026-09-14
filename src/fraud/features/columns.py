"""Resolve a feature config into concrete column lists.

Pure: takes the config and the frame's columns, returns names. Nothing is fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from fraud.data import schema

IDENTITY_NUMERIC: tuple[str, ...] = tuple(
    c for c in schema.IDENTITY_COLS if c != schema.ID_COL and c not in schema.IDENTITY_STR_COLS
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    target: str
    id_col: str
    time_col: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    derived: tuple[str, ...] = ()
    frequency: tuple[str, ...] = ()

    @property
    def all_inputs(self) -> tuple[str, ...]:
        return self.numeric + self.categorical + self.frequency


def load_feature_spec(path: Path) -> FeatureSpec:
    raw = yaml.safe_load(path.read_text())
    num = raw["numeric"]
    numeric: list[str] = list(num.get("explicit", []))
    for prefix in num.get("prefixes", []):
        family = {"C": schema.C_COLS, "D": schema.D_COLS, "V": schema.V_COLS}[prefix]
        numeric.extend(family)
    if num.get("identity_numeric", False):
        numeric.extend(IDENTITY_NUMERIC)
    numeric.extend(num.get("boolean", []))
    categorical = tuple(str(c) for c in raw.get("categorical", []))
    derived = tuple(str(d) for d in raw.get("derived", []))
    frequency = tuple(str(c) for c in raw.get("frequency", []))
    spec = FeatureSpec(
        name=str(raw["name"]),
        target=str(raw["target"]),
        id_col=str(raw["id_col"]),
        time_col=str(raw["time_col"]),
        numeric=tuple(numeric),
        categorical=categorical,
        derived=derived,
        frequency=frequency,
    )
    forbidden = {spec.target, spec.id_col, spec.time_col}
    leaked = forbidden & set(spec.all_inputs)
    if leaked:
        raise ValueError(f"feature spec {spec.name} uses forbidden columns {sorted(leaked)}")
    raw_inputs = spec.numeric + spec.categorical
    dupes = [c for c in set(raw_inputs) if raw_inputs.count(c) > 1]
    if dupes:
        raise ValueError(f"feature spec {spec.name} lists columns twice: {sorted(dupes)}")
    return spec
