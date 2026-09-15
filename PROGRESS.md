# Progress

Stage checklist. Each stage is a working system before the next begins.

## Stage 0 — Scaffolding
- [x] Repo initialised, layout, `pyproject.toml`, `uv.lock`
- [x] GitHub remote (public)
- [x] Dataset downloaded into `data/raw/`
- [x] Raw-data loader + schema checks + data tests

## Stage 1 — Data understanding
- [x] EDA script (`scripts/eda.py` → `reports/eda/`)
- [x] `docs/eda.md`: distributions, missingness, temporal structure, label rate over time
- [x] Problem formulation ADR (what `isFraud` means; what is being predicted)

## Stage 2 — Temporal split, leakage analysis, baselines
- [x] Split-strategy ADR + `configs/split.yaml` + split tests
- [x] Primary-metric ADR (also fixes imbalance handling: none by default)
- [x] `docs/leakage_audit.md` for the first feature families
- [x] Constant baseline + logistic-regression pipeline, logged to MLflow (E001 val PR-AUC 0.034, E002 0.402)

## Stage 3 — Feature engineering + XGBoost
- [x] Feature ADRs and leakage audit rows per family (ADR 0005 frequency encoding accepted as E006; hour/weekday rejected as E005)
- [x] XGBoost pipeline beats baselines on validation (E003 val PR-AUC 0.570, gap 0.22); seed noise measured (E004, sd 0.002)
- [ ] Tuning run compared to untuned

## Stage 4 — Threshold policy and calibration
- [x] Cost-model ADR, threshold document (threshold 0.08; `docs/threshold.md`)
- [x] Calibration evaluated (reliability diagram, Brier); sigmoid applied (ADR 0007)

## Stage 5 — Ablation, importance, model card
- [ ] Ablation per feature group; importance from two methods
- [ ] `docs/model_card.md`; single test-window evaluation

## Stage 6 — Serving
- [ ] FastAPI `/health` + `/predict`; parity test with offline pipeline
- [ ] Dockerfile (non-root, from lock file)
- [ ] Small UI; deployment or documented local-only reason
- [ ] README complete and verified from a clean clone

## Current stage: **3 — Feature engineering + XGBoost**
ADR 0004 written; E007 entity history rejected (no gain when restricted to legitimate computation).
Next: Stage 4 — calibrate, choose the threshold, learning curves.
