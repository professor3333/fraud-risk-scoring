# Progress

Stage checklist. Each stage is a working system before the next begins.

## Stage 0 — Scaffolding
- [x] Repo initialised, layout, `pyproject.toml`, `uv.lock`
- [ ] GitHub remote (public)
- [ ] Dataset downloaded into `data/raw/`
- [ ] Raw-data loader + schema checks + data tests

## Stage 1 — Data understanding
- [ ] EDA notebook (exploration only)
- [ ] `docs/eda.md`: distributions, missingness, temporal structure, label rate over time
- [ ] Problem formulation ADR (what `isFraud` means; what is being predicted)

## Stage 2 — Temporal split, leakage analysis, baselines
- [ ] Split-strategy ADR + `configs/split.yaml` + split tests
- [ ] Primary-metric ADR
- [ ] `docs/leakage_audit.md` for the first feature families
- [ ] Constant baseline + logistic-regression pipeline, logged to MLflow

## Stage 3 — Feature engineering + XGBoost
- [ ] Feature ADRs and leakage audit rows per family
- [ ] XGBoost pipeline beats baselines on validation (gap reported)
- [ ] Tuning run compared to untuned

## Stage 4 — Threshold policy and calibration
- [ ] Cost-model ADR, threshold document
- [ ] Calibration evaluated (reliability diagram, Brier); applied only if needed

## Stage 5 — Ablation, importance, model card
- [ ] Ablation per feature group; importance from two methods
- [ ] `docs/model_card.md`; single test-window evaluation

## Stage 6 — Serving
- [ ] FastAPI `/health` + `/predict`; parity test with offline pipeline
- [ ] Dockerfile (non-root, from lock file)
- [ ] Small UI; deployment or documented local-only reason
- [ ] README complete and verified from a clean clone

## Current stage: **0 — Scaffolding**
Next: create GitHub remote, download data, write the loader and data tests.
