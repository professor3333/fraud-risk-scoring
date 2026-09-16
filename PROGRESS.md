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
- [x] Ablation per feature group; importance from two methods (`docs/ablation.md`)
- [x] `docs/model_card.md`; single test-window evaluation (test PR-AUC 0.553)

## Stage 6 — Serving
- [x] FastAPI `/health` + `/predict`; parity test with offline pipeline (fixture + slow real-data test)
- [x] Dockerfile (non-root, from lock file; 1.5 GB image, runtime deps only)
- [x] Small UI (demo page at `/`); local-only, reason in README
- [x] README complete and verified from a clean clone (v0.1.0 tagged)

## Current stage: **3 — Feature engineering + XGBoost**
ADR 0004 written; E007 entity history rejected (no gain when restricted to legitimate computation).
Next: Stage 4 — calibrate, choose the threshold, learning curves.

## Completion evidence

| area | evidence |
|---|---|
| Data | `scripts/download_data.py`, `scripts/validate_data.py`, validated left join, 18 data tests |
| EDA | `scripts/eda.py` → `reports/eda/`; conclusions in `docs/eda.md` |
| Validation | ADR 0002 (frozen split), ADR 0008 (rolling backtests with horizons, `docs/backtest.md`); E010 random-split control |
| Leakage | `docs/leakage_audit.md` |
| Baseline | E001 dummy, E002b / E002 logistic regression |
| Features | F0 → F7 (E005 – E017), ladder + family cuts (E018 – E020), `docs/feature_sets.md` |
| Model | `scripts/train.py` from config, seeds logged, reproducibility test; E022 shipped and confirmed by E023; monthly lifecycle `scripts/retrain.py` (E024) |
| Imbalance | ADR 0003 |
| Evaluation | PR-AUC, ROC-AUC, threshold metrics, top-k/day, Brier/ECE; `reports/final/`; subgroup robustness (`docs/subgroups.md`, `reports/subgroups/`) |
| Experiments | MLflow (`sqlite:///mlflow.db`), provenance hashes per run |
| Reasoning | `docs/EXPERIMENT_LOG.md`, `docs/experiments.md` |
| Operations | threshold 0.08 (ADR 0006), review policy (`docs/review_policy.md`); delayed-label loop — outcomes → predictions → cohort-aged monitoring → retraining on matured labels (ADR 0009, `docs/feedback.md`, `reports/feedback/`) |
| Artifact | one calibrated object + frozen golden, startup parity (`fraud.serve.parity`); promotion gates → `models/champion/` + MLflow registry alias (ADR 0010, `docs/promotion.md`) |
| API | `/health`, `/predict`, `/predict/batch`, `/predict/csv`, `/model-info` |
| UI | analyst dashboard at `/` |
| Quality | 93 tests (`docs/testing.md`), CI green with container check |
| Deployment | Docker image; Fly.io config + `scripts/deploy_check.py` (`docs/deployment.md`) — **public endpoint pending the account login** |
| Documentation | README, ADRs 0001–0010, model card, error analysis, progression |
| Learning | `docs/defending_the_decisions.md` — the owner's study sheet |
