"""Feature-group ablation and importance from two methods (§8): gain and group permutation."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from fraud.data import schema
from fraud.features.columns import FeatureSpec

GROUP_PATTERNS: dict[str, str] = {
    "amount": r"^TransactionAmt$",
    "product": r"^ProductCD$",
    "card": r"^card[1-6]$",
    "addr": r"^addr[12]$",
    "dist": r"^dist[12]$",
    "email": r"^[PR]_emaildomain$",
    "C": r"^C\d+$",
    "D": r"^D\d+$",
    "M": r"^M\d$",
    "V": r"^V\d+$",
    "identity": r"^(id_\d\d|DeviceType|DeviceInfo)$",
    "has_identity": rf"^{schema.HAS_IDENTITY_COL}$",
    "frequency": r"^freq_",
}


def group_of(column: str) -> str:
    for name, pattern in GROUP_PATTERNS.items():
        if re.match(pattern, column):
            return name
    return "other"


def spec_without_group(spec: FeatureSpec, group: str) -> FeatureSpec:
    """Drop every raw and frequency-encoded column belonging to ``group``."""
    if group == "frequency":
        return replace(spec, frequency=())
    keep = [c for c in spec.numeric if group_of(c) != group]
    keep_cat = [c for c in spec.categorical if group_of(c) != group]
    keep_freq = [c for c in spec.frequency if group_of(c) != group]
    return replace(
        spec, numeric=tuple(keep), categorical=tuple(keep_cat), frequency=tuple(keep_freq)
    )


def source_column(feature_name: str, spec: FeatureSpec) -> str:
    """Map a pipeline output name back to the raw column it came from."""
    prefix, raw = feature_name.split("__", 1)
    if prefix == "freq":
        return raw.removeprefix("freq_")
    raw = re.sub(r"^missingindicator_", "", raw)
    if prefix == "cat":  # one-hot: '<column>_<level>'; pick the longest matching column
        matches = [c for c in spec.categorical if raw == c or raw.startswith(c + "_")]
        if matches:
            return max(matches, key=len)
    return raw


def gain_importance(pipe: Pipeline, spec: FeatureSpec) -> pd.DataFrame:
    """Per-feature gain from the booster, with the pipeline's output names and their group."""
    model = pipe.named_steps["model"]
    names = list(pipe.named_steps["features"].get_feature_names_out())
    gain = model.get_booster().get_score(importance_type="gain")
    rows = []
    for i, name in enumerate(names):
        source = source_column(name, spec)
        group = "frequency" if name.startswith("freq__") else group_of(source)
        rows.append(
            {
                "feature": name,
                "source": source,
                "gain": float(gain.get(f"f{i}", 0.0)),
                "group": group,
            }
        )
    df = pd.DataFrame(rows)
    df["gain_share"] = df["gain"] / df["gain"].sum()
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


def group_permutation_importance(
    pipe: Pipeline,
    frame: pd.DataFrame,
    target: str,
    spec: FeatureSpec,
    seed: int,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Drop in validation PR-AUC when a whole raw-column group is shuffled together.

    Shuffling a group jointly keeps within-group structure (the ``V`` blocks are
    highly correlated) and asks the question the ablation asks, without refitting.
    """
    rng = np.random.default_rng(seed)
    y = frame[target].to_numpy()
    base = float(average_precision_score(y, pipe.predict_proba(frame)[:, 1]))
    groups: dict[str, list[str]] = {}
    for col in spec.numeric + spec.categorical + spec.frequency:
        groups.setdefault(group_of(col), []).append(col)
    rows = []
    for group, cols in groups.items():
        cols = sorted(set(cols))
        drops = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            perm = rng.permutation(len(frame))
            shuffled[cols] = frame[cols].to_numpy()[perm]
            drops.append(
                base - float(average_precision_score(y, pipe.predict_proba(shuffled)[:, 1]))
            )
        rows.append(
            {
                "group": group,
                "n_columns": len(cols),
                "pr_auc_drop_mean": float(np.mean(drops)),
                "pr_auc_drop_std": float(np.std(drops)),
            }
        )
    out = pd.DataFrame(rows).sort_values("pr_auc_drop_mean", ascending=False)
    out.attrs["base_pr_auc"] = base
    return out.reset_index(drop=True)
