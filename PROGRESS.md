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
- [x] Tuning run compared to untuned (`docs/tuning.md`: +0.036 on the same features, then +0.017 seed-paired)

## Stage 4 — Threshold policy and calibration
- [x] Cost-model ADR, threshold document (threshold 0.08; `docs/threshold.md`)
- [x] Calibration evaluated (reliability diagram, Brier); sigmoid applied (ADR 0007)

## Stage 5 — Ablation, importance, model card
- [x] Ablation per feature group; importance from two methods (`docs/ablation.md`)
- [x] `docs/model_card.md`; test-window evaluation (E022: test PR-AUC 0.561; reporting window, not a blind holdout — consultation log in ADR 0002)

## Stage 6 — Serving
- [x] FastAPI `/health` + `/predict`; parity test with offline pipeline (fixture + slow real-data test)
- [x] Dockerfile (non-root, from lock file; 1.5 GB image, runtime deps only)
- [x] Small UI (analyst dashboard at `/`); public at https://fraud-risk-scoring-m1fp.onrender.com
- [x] README complete and verified from a clean clone (first tagged at v0.1.0; v0.8.0 current)

## Current stage: **all six complete — operating and hardening**
Every stage box above is ticked. Work since v0.5.0 is post-stage
operations: label feedback (ADR 0009), promotion gates (ADR 0010),
subgroup robustness, upload limits, the API benchmark, the per-day review
budget. Per-prediction explanation added (`docs/explanation.md`). API key, time budget and structured request logs added
(`docs/deployment.md` → Hardening). Per-client application rate limits added for single
predictions, CSV/batch uploads and explanations. CD on tag exists (`deploy.yml` + `scripts/release.py`). The hosting
target moved from Fly.io (paid) via Hugging Face Spaces (Docker now needs
PRO) to a Render free web service (512 MB / 0.1 CPU, no card; the service
fits at 238 MiB). **Live since 2026-09-17: https://fraud-risk-scoring-m1fp.onrender.com**
(`deploy_check.py` passes against it; the champion is served from its
GitHub release). Continuous deployment on tag is configured through `RENDER_DEPLOY_HOOK`
and `RENDER_URL`; the workflow waits for the new version and verifies the live service. A documentation scope pass (2026-09-18) states plainly what is
operated versus simulated: retraining runs manually, the free
demo's SQLite state is ephemeral, and autonomous blocking is a demonstration
policy whose block-precision target does not hold in every subgroup. MIT
`LICENSE` added and recorded in `pyproject.toml`. **Monitoring is now
scheduled** (2026-09-21, ADR 0011): the service builds its own report over its
audit trail (`GET /audit/monitor`), a daily GitHub Action fetches it and opens
one labelled issue per alert episode, closing it on the first healthy run.
Retraining is still the operator's call — nothing promotes automatically.

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
| API | `/health`, `/predict`, `/predict/batch`, `/predict/csv`, `/explain`, `/model-info`, `/audit/recent`, `/audit/monitor`, `/outcomes`; one `action` per row, per-day review budget shared across requests through the audit trail; `X-API-Key` when `FRAUD_API_KEY` is set |
| UI | analyst dashboard at `/` |
| Quality | 183 fixture tests + 4 slow (`docs/testing.md`), CI green with container check |
| Deployment | Docker image; free public target = Render web service built without weights, fetching the champion from its `champion-<sha>` GitHub release at startup (published) (measured under Render's limits locally, CD-wired); Fly.io as the paid alternative; `scripts/deploy_check.py` (`docs/deployment.md`) — **live at https://fraud-risk-scoring-m1fp.onrender.com** |
| Monitoring | frozen reference per artifact, report over any window (`scripts/monitor.py`, local or `--from-url`), scheduled daily against the live service with GitHub-issue alerts (ADR 0011, `monitor.yml`, `scripts/alert.py`, `configs/alerting.yaml`, `docs/monitoring.md`) |
| Security | digest-pinned champion, commit-pinned deploy, Dependabot (uv / actions / docker), weekly `pip-audit` + CodeQL (`security.yml`); scope and gaps in `docs/security.md` |
| Documentation | MIT `LICENSE`, README, ADRs 0001–0011, model card, error analysis, explanation, subgroups, monitoring, feedback, promotion, retraining, deployment, progression |
| Learning | `docs/defending_the_decisions.md` — the owner's study sheet |
