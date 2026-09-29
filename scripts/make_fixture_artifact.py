"""Write a stand-in served artifact trained on the test fixture (for CI's Docker build).

Not a real model: it exists so the Dockerfile can be built and smoke-tested
where the labelled data and the trained artifact are not available.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import yaml

from fraud.data.load import load_train
from fraud.data.split import load_split_config, split
from fraud.features.columns import load_feature_spec
from fraud.monitor.reference import build_reference, save_reference
from fraud.pipeline.build import build_pipeline
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.parity import artifact_digest, choose_sample, freeze

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "champion" / "model.joblib"  # no manifest: serving.yaml's fallbacks apply


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
    # The monitoring reference too, so the container smoke test can build a real
    # /audit/monitor report inside the runtime image rather than stop at a 404.
    serving = yaml.safe_load((ROOT / "configs" / "serving.yaml").read_text())
    reference = build_reference(
        model,
        parts["train"],
        held,
        float(serving["bands"]["block"]),
        int(serving.get("default_review_budget", 200)),
        artifact_digest(OUT),
    )
    save_reference(reference, OUT)
    print(f"wrote fixture-trained stand-in artifact to {OUT} (+ frozen sample, monitor reference)")


if __name__ == "__main__":
    main()
