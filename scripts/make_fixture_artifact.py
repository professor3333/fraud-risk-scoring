"""Write a stand-in served artifact trained on the test fixture (for CI's Docker build).

Not a real model: it exists so the Dockerfile can be built and smoke-tested
where the labelled data and the trained artifact are not available.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.features.columns import load_feature_spec
from fraud.pipeline.build import build_pipeline
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.parity import choose_sample, freeze

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "xgb_f5_capacity_calibrated.joblib"


def main() -> None:
    if OUT.exists():
        print(f"{OUT} exists; leaving it alone")
        return
    parts = split(
        load_train(ROOT / "tests" / "fixtures" / "raw"),
        load_split_config(ROOT / "configs" / "split.yaml"),
    )
    spec = load_feature_spec(ROOT / "configs" / "features" / "v2_freq.yaml")
    pipe = build_pipeline(spec, {"type": "xgboost", "params": {"n_estimators": 30}}, seed=0)
    pipe.fit(parts["train"], parts["train"][spec.target])
    held = parts["validation"]
    model = CalibratedModel(pipe, method="sigmoid").fit_calibrator(
        pipe.predict_proba(held)[:, 1], held[spec.target]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    freeze(model, OUT, choose_sample(parts["test"], n_per_group=5))
    print(f"wrote fixture-trained stand-in artifact to {OUT} (+ frozen sample)")


if __name__ == "__main__":
    main()
