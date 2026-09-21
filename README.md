# Fraud Risk Scoring

[![ci](https://github.com/professor3333/fraud-risk-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/professor3333/fraud-risk-scoring/actions/workflows/ci.yml)

A fraud risk scoring system built on the Kaggle IEEE-CIS Fraud Detection
dataset. Given one online transaction, it returns a calibrated probability
that the transaction is fraudulent and the risk band that probability falls
in. Given a batch or a CSV, it also returns one action per row — approve /
review / block — from a review policy sized to an analyst budget, because the
review half of that policy ranks a transaction against the rest of its day and
so cannot be decided for a lone request. Served over a FastAPI endpoint, with
every modelling decision recorded and every improvement measured on a strictly
later window of time.

**Live demo:** https://fraud-risk-scoring-m1fp.onrender.com — the analyst
dashboard; `https://fraud-risk-scoring-m1fp.onrender.com/docs` is the API.
A Render free web service (512 MB, 0.1 CPU) that fetches the model from
its GitHub release at startup: it sleeps after 15 minutes idle, so the
first request after a pause takes about a minute, and the 200-row sample
scores in ~20 s at a tenth of a CPU. A demo, not a service
(`docs/deployment.md`).

> **Demo only — do not upload real cardholder, customer, or confidential
> transaction data.** Use the included synthetic sample or synthetic data of your
> own. Rows you submit are scored and written to the service's prediction audit
> trail, which is how the monitoring here is demonstrated.

**Public demo persistence is ephemeral; durable auditing/feedback requires persistent storage.**
On Render, a restart or redeploy resets the prediction audit trail, any stored
delayed labels, cross-request review-budget accounting, and locally stored
monitoring history.

## Architecture

```
                 raw CSVs (Kaggle)                         request (JSON, one transaction)
                        │                                              │
            fraud.data.load  ──  validate schema,                      │
            join identity, has_identity flag                           │
                        │                                              │
            fraud.data.split ── temporal windows                       │
            train 1–122 · validation 123–152 · test 153–183            │
                        │                                              │
   ┌────────────────────┴────────────────────┐            fraud.serve.app
   │  one sklearn Pipeline (fit on train only) │◄────────  loads the SAME object
   │   Derive → ColumnTransformer → XGBoost    │           once at startup
   │   · one-hot low-cardinality (+<missing>)  │                       │
   │   · frequency tables (train population)   │                       ▼
   │   · numerics with NaN kept                │            /predict → probability,
   └────────────────────┬────────────────────┘             risk level,
                        │                                   model version
                        │                                  /predict/batch, /predict/csv
                        │                                   → the above + one action
        CalibratedModel (sigmoid map fit on
        out-of-fold training-window scores)
                        │
   MLflow: params, split, features, metrics, curves, artifact
   evaluation threshold 0.08 (cost curve on validation); served policy:
   block ≥ 0.42 · review top-N/day by analyst budget · approve the rest
```

Design rules that shaped it (all enforced by tests):

- **Time is respected.** Splits are by `TransactionDT`; validation and test are
  strictly later than training; tuning folds are expanding windows inside
  the training window; every fitted statistic (one-hot vocabularies,
  frequency tables, calibration map) is learned from training rows only.
- **One preprocessing path.** The object that produced the validation
  metrics is the object the API loads. A frozen 50-row golden is checked at
  service startup and in tests: training pipeline == saved artifact == API
  == stored probabilities, or the service refuses to start
  (`fraud.serve.parity`).
- **Every change is an experiment.** Hypothesis written first, one change at
  a time, logged to MLflow, accepted only if it clears a measured noise floor
  (`docs/experiments.md`).
- **Assumptions are tests.** Identity coverage, window ordering, "nothing fit
  on validation", strictly-earlier-rows features, save/load parity,
  reproducibility.

## Problem

Card-not-present fraud on an e-commerce payment stream. At authorization,
should this transaction be approved or declined? Positives are 3.5 % of
transactions; a missed fraud costs the amount plus a chargeback fee, a
false decline costs a customer and some margin. The label is the provider's
`isFraud`, which — per the competition host — marks the reported
transaction *and* subsequent transactions on the same account, so "fraud"
here means "belongs to an account that becomes reported" (ADR 0001). Full
problem definition, split, metric, feature, threshold and calibration
decisions are in `docs/decisions/`.

## Results

Validation window = days 123 – 152 (85,044 transactions, 2,884 fraud), used
for every decision. Test window = days 153 – 183 (85,430 / 2,994), the
**final temporal reporting window**: no model was selected on it, but three
candidates and two policy checks were reported on it during development, so
it is not a strictly blind holdout (consultation log in ADR 0002).

| experiment | model | val PR-AUC | val ROC-AUC | recall @ P ≥ 0.90 |
|---|---|---:|---:|---:|
| E001 | constant prior | 0.034 | 0.500 | 0.000 |
| E002b | logistic regression, 26 interpretable columns | 0.288 | 0.795 | 0.082 |
| E002 | logistic regression, all 423 raw columns | 0.402 | 0.842 | 0.118 |
| E003 | XGBoost, raw columns | 0.570 | 0.912 | 0.292 |
| E005 | + hour / weekday | 0.576 · rejected (+0.002 seed-paired) | 0.912 | 0.284 |
| E006 | + frequency encoding (ADR 0005) | 0.578 · accepted (+0.009 seed-paired) | 0.919 | 0.294 |
| E007 | + card-history features, earlier rows only (ADR 0004) | 0.582 · rejected (−0.002 seed-paired; see E025) | 0.920 | 0.280 |
| E008 | E006 tuned by expanding-window CV | 0.616 | 0.929 | 0.330 |
| E011 | E008 + median imputation & indicators before the trees | 0.613 · neutral | 0.933 | 0.326 |
| E012 | E008 + rare one-hot levels grouped | 0.621 · neutral (+0.002 seed-paired) | 0.930 | 0.326 |
| E013–E015 | E008 + missing counts / amount structure / e-mail families | 0.615 / 0.615 / 0.610 · rejected | | |
| **E016** | E008 + card × address / e-mail keys, frequency-encoded | **0.619** (+0.006 seed-paired) | **0.932** | 0.329 |
| E017 | E016 + card+address history, earlier rows only | 0.619 · rejected | 0.931 | 0.329 |
| E018–E020 | cumulative ladder + family cuts; F0–F6 cumulative and no-`V` seed-paired | 0.626→+0.000, 0.621→−0.003 · neutral | | |
| E021 | one-axis sweeps around E016 | validation never turns over: depth 12 0.629, train-1.000 model 0.627 | | |
| **E022** | depth 12, min_child_weight 1, 1,600 trees | **0.637** (+0.017 seed-paired) | **0.933** | **0.334** |

Final model (E022 + sigmoid calibration, threshold 0.08):

| metric | validation | test |
|---|---:|---:|
| PR-AUC | 0.637 | **0.561** |
| ROC-AUC | 0.933 | 0.907 |
| precision / recall / F1 at 0.08 | 0.346 / 0.743 / 0.472 | 0.262 / 0.691 / 0.380 |
| recall reviewing the top 500 transactions per day | 0.873 | 0.801 |
| Brier (prior 0.033) / ECE | 0.0185 / 0.0037 | 0.0214 / 0.0059 |
| cost at 0.08 vs approve-all | 218k vs 487k (−55 %) | 275k vs 481k (−43 %) |

The previous shipped model (E016) scored 0.619 / 0.557. E022's +0.017 on
validation became +0.004 on the test month, so the validation protocol was
extended (ADR 0008, `docs/backtest.md`): rolling backtests with training
cut-offs at days 60 / 90 / 120 scored 0–60 days out. Under it E022's edge
over E016 halves with horizon (+0.013 adjacent → +0.006 at 60 days) but
holds on every one of ten windows and every seed (+0.007 paired), so it
stands on validation-only evidence. The same backtests put a number on
staleness: a model one month older loses ~0.1 PR-AUC on the next month,
and the offline retraining lifecycle (`scripts/retrain.py`,
`docs/retraining.md`) promotes a fresh challenger by +0.125.

A control experiment (E010) trains the same model on a random stratified
split of the same rows: it reports validation PR-AUC **0.810** against the
temporal split's 0.616 — the random split hides both drift and the
account-level label propagation, and would have promised performance the
next month never delivers.

**How much is Vesta's engineering and how much is mine?** (`docs/feature_sets.md`)
With the tuned parameters on the frozen validation split: the provider's
raw columns reach 0.601; my frequency tables and composite keys add ≈ 0.02
wherever they are added (0.619 shipped); removing Vesta's `V` block from
the shipped set costs nothing (0.621, seed-paired −0.003 — a 100-input
model at the same score); removing all of Vesta's engineered families
(`C`, `D`, `M`, `V`) costs **0.19** (0.433). The provider's history
summaries carry the model; the work here adds a few points on top and shows
the largest provider block is optional once entity frequency is modelled.

**What fools the model** (`docs/error_analysis.md`, from `reports/final/`).
The confident false positives are the fraud archetype performed by real
customers — new card, product `C`, no billing address, self-addressed
e-mail — indistinguishable row by row (hence review, not block). The
confident false negatives are fraud that looks like everyone else: two
thirds share an account with other labelled fraud (label propagation, not
recoverable at authorization) and the rest are established cards used for
the mainstream product (recoverable only through per-entity deviation
features on that slice).

**From probability to action** (`docs/review_policy.md`). With three
actions — block when the calibrated probability is ≥ 0.42 (four in five
such transactions are fraud on validation), review the next-highest scores
down to the analyst budget, approve the rest — a 200-review/day team
catches 77 % of fraud at a validation cost 30 % below the best single
decline threshold and 67 % below approving everything. One month later the
ranking holds (75 % recall at the same budget) while the probability scale
drifts (block precision 0.80 → 0.71), so the recommended deployment rule is
threshold-based blocking and **rank-based** reviewing.

Feature engineering was run as named sets, one experiment each: missing
counts, amount structure, time of day, e-mail families and two definitions
of card history all added nothing (E005, E007, E013 – E015, E017); only
frequency encoding (E006) and frequency-encoded card × address / e-mail
keys (E016) earned their place. The reconstructed card identifier that
drives Kaggle solutions adds nothing once restricted to what a live system
can compute — **with one qualifier added later**: every one of those
feature experiments ran at the capacity shipped at the time, and when the
card-history set was re-run at the deeper model's capacity it turned
positive (E025: +0.0066 overall, +0.043 on cards first seen under a
fortnight ago). It is measured, written up and **not shipped** — serving it
requires the history store ADR 0004 makes binding. Ablation shows the 339 `V`
columns carry 76 % of split gain but are almost fully substitutable
(−0.005 when removed), while `card*` and `C*` are irreplaceable
(`docs/ablation.md`). Interpretation, limitations and drift are in
`docs/model_card.md`.

## Features (what exists)

- Raw-data contract and loader with join validation and a Parquet cache.
- Temporal split from YAML; dev config with a seeded 10 % sample.
- sklearn pipeline with in-pipeline derived features, `<missing>` one-hot
  levels, and frequency encoding fit on the training window.
- Training CLI logging git commit, data / split / feature-set / config
  content hashes, all hyperparameters, the full metric set, training time,
  PR and ROC curves, confusion matrix, gain importance and the fitted
  pipeline to MLflow; reasoning in `docs/EXPERIMENT_LOG.md`.
- Expanding-window random search; one-axis sweeps; post-hoc learning
  curves; rolling temporal backtests with forecast horizons (ADR 0008).
- Monthly retraining lifecycle simulated offline: retrain → calibrate →
  re-select block threshold → champion/challenger → promote → freeze.
- Sigmoid / isotonic calibration on out-of-fold scores, chosen by rule.
- Amount-weighted cost curve, sensitivity table, train-only cross-check;
  three-action review policy sized to an analyst budget.
- Gain, group-permutation and group-ablation importance; subgroup
  robustness table (product, card, identity, e-mail family, device, amount,
  time) under the served policy (`docs/subgroups.md`); per-prediction
  explanation from XGBoost's own TreeSHAP contributions, summed per source
  column and family with the calibration step stated (`POST /explain`,
  shown in the dashboard's row inspector; `docs/explanation.md`).
- Single test-window evaluation with top-*k*-per-day review metrics.
- FastAPI service: `/health`, `/predict` (probability + risk band),
  `/predict/batch` and `/predict/csv` (the same, plus one authoritative
  `action` — approve / review / block — from the rank-based review policy,
  which needs a day's transactions to rank against), `/explain`,
  `/model-info`, `/audit/recent`,
  `/audit/monitor`; startup parity
  check against a frozen golden; every scored transaction, every request
  and an input snapshot persisted to a SQLite audit trail; two-tier access
  (`FRAUD_API_KEY` for scoring, `FRAUD_ADMIN_API_KEY` for `/outcomes` and
  `/audit/*`, which stay closed without it), upload cap, per-request time
  budget, request IDs and one JSON log line per call for the public URL
  (`docs/deployment.md`).
- Monitoring: a frozen reference per artifact and a report over any window
  of stored predictions — API rate / latency / errors, score and action
  PSI, per-feature drift, eventual performance and calibration with labels;
  the service builds the same report over its own trail (`GET /audit/monitor`)
  and a daily GitHub Action fetches it and raises one issue per alert episode
  (ADR 0011).
- Automated retraining: a monthly cycle decides whether a month of matured
  labels the champion has not seen exists, fits a challenger, scores it and
  the champion on that month, applies the gates plus a margin, promotes,
  publishes the artifact and opens the policy change as a pull request
  (ADR 0012); merging it deploys and verifies the digest.
- Model promotion: candidate → acceptance gates (PR-AUC, recall at the
  review budget, ECE, Brier, block precision, cost per transaction, parity)
  → champion; the MLflow model registry is the ledger, `models/champion/`
  is what the service and the image load, its manifest supplies the
  version, facts and policy bands (ADR 0010, `docs/promotion.md`).
- Delayed-label feedback loop: outcomes arrive later (`POST /outcomes` or
  a simulated feed), attach to their predictions, age against the 120-day
  reporting window (matured / pending / overdue); eventual metrics on
  closed cohorts only, an early signal on open ones, retraining that
  waits for maturity (ADR 0009, `docs/feedback.md`).
- Analyst dashboard at `/`: CSV upload → summary tiles, ranked table with
  sort / filter / search, row inspector, ranked-CSV download; single-
  transaction form at `/single`. Dockerfile.
- 153 fixture-based tests (no data, no network) + 4 slow real-data tests.

## Tech stack

Python 3.12 · uv · pandas 3 · scikit-learn · XGBoost 3 · MLflow 3 (SQLite)
· FastAPI + Pydantic + uvicorn · a dependency-free HTML/JS dashboard ·
matplotlib · pytest · ruff · mypy · Docker.

## Project structure

```
configs/          split.yaml, dev.yaml, features/*.yaml, model/*.yaml,
                  tuning/*.yaml, backtest.yaml, retrain.yaml, threshold.yaml,
                  policy.yaml, serving.yaml, feedback.yaml, promotion.yaml,
                  alerting.yaml
data/             git-ignored; data/README.md explains the download
docs/             eda.md, decisions/ (ADR 0001–0012), EXPERIMENT_LOG.md (4-column
                  ledger), experiments.md (long form), leakage_audit.md,
                  threshold.md, review_policy.md, ablation.md, feature_sets.md,
                  error_analysis.md, xgboost_progression.md, testing.md,
                  backtest.md, retraining.md, deployment.md, monitoring.md,
                  feedback.md, promotion.md, subgroups.md, explanation.md,
                  tuning.md, defending_the_decisions.md, model_card.md
reports/          committed evidence: EDA figures, curves, calibration,
                  threshold, policy, ablation, feature sets, test, final
scripts/          download_data, validate_data, eda, train, tune, param_sweep,
                  backtest, retrain, learning_curve, calibrate, select_threshold,
                  review_policy, freeze_artifact, monitor_reference, monitor,
                  feedback, simulate_feedback, promote, subgroups, ablation,
                  feature_ladder, evaluate_test, final_report, benchmark_api,
                  release, publish_champion, split_comparison, deploy_check,
                  alert, retrain_cycle, make_fixture_artifact
deploy/hosted/    Dockerfile without the weights (Render builds it; the service
                  fetches the champion from its release at startup)
src/fraud/
  data/           schema (contract), validate (checks), load (read + join), split
  features/       columns (spec), derive, time, rowwise (F1/F2/F4/F5), encoders, history
  pipeline/       build (preprocessing + model), calibrated
  train/          run (MLflow), tune (expanding-window CV), lifecycle (offline
                  replay), trigger + cycle (one scheduled cycle), promotion,
                  serving_config (the policy patch a promotion implies)
  evaluate/       metrics, curves, calibration, threshold, importance
  serve/          app, schemas, frames, parity (frozen-golden check), audit
                  (SQLite events, requests, input snapshots), static/
  monitor/        drift (PSI, bins), reference (frozen 'normal'), report,
                  remote (fetch a deployed service's report), alerts (what pages)
tests/            fixtures/ (synthetic 400-row raw files + generator), test_*.py
.github/          ci.yml, security.yml, deploy.yml (tag → Render), monitor.yml
                  (daily report on the live service → issue on alert),
                  retrain.yml (monthly cycle → PR), deploy-champion.yml
                  (merged policy change → the host serves it)
Dockerfile        runtime-only image, non-root
render.yaml       Render blueprint — the free public demo (the live deployment)
fly.toml          Fly.io app definition — the optional paid alternative
```

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/).
- For training: the Kaggle dataset (≈ 1.3 GB extracted) and ~8 GB RAM.
- For the Docker image: Docker 24+.

## Installation

```bash
git clone https://github.com/professor3333/fraud-risk-scoring.git
cd fraud-risk-scoring
uv sync
uv run pytest            # 200 fixture tests; no data/network required
```

## Usage

Everything below was run as written.

**Get the data** (needs a Kaggle account that has accepted the competition
rules and an API token; see `data/README.md`):

```bash
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
uv run python scripts/download_data.py
uv run python scripts/validate_data.py     # contract report: columns, dtypes, ids, label, join, row counts
```

**Reproduce the model** (from raw CSVs to the served artifact):

```bash
uv run python scripts/eda.py                                                  # ~20 s: summary JSON, CSVs, figures
uv run python scripts/train.py --model configs/model/constant.yaml            # E001
uv run python scripts/train.py --model configs/model/logreg_small.yaml        # E002b, ~20 s
uv run python scripts/train.py --model configs/model/logreg.yaml              # E002, ~16 min
uv run python scripts/train.py --model configs/model/xgboost.yaml             # E003, ~45 s
uv run python scripts/train.py --model configs/model/xgboost_v2_freq.yaml     # E006
uv run python scripts/tune.py  --config configs/tuning/xgboost.yaml           # E008 search, ~2 h on 8 GB
uv run python scripts/train.py --model configs/model/xgboost_v2_tuned.yaml    # E008 refit, ~90 s
uv run python scripts/train.py --model configs/model/xgboost_f5_interactions.yaml  # E016
uv run python scripts/param_sweep.py --config configs/tuning/sweep.yaml           # E021, ~1 h
uv run python scripts/train.py --model configs/model/xgboost_f5_capacity.yaml      # E022, shipped, ~4 min
uv run python scripts/backtest.py --candidates configs/model/xgboost_f5_capacity.yaml  # ADR 0008 horizons, ~15 min
uv run python scripts/retrain.py                                               # offline monthly simulation, ~20 min
uv run python scripts/learning_curve.py --model models/xgb_v2_tuned.joblib
uv run python scripts/calibrate.py --model-config configs/model/xgboost_f5_capacity.yaml
uv run python scripts/select_threshold.py --run-name xgb_f5_capacity
uv run python scripts/review_policy.py --run-name xgb_f5_capacity --with-test
uv run python scripts/freeze_artifact.py --run-name xgb_f5_capacity            # frozen golden for parity
uv run python scripts/promote.py --run-name xgb_f5_capacity                    # gates -> models/champion/ (served)
uv run python scripts/monitor_reference.py --run-name xgb_f5_capacity          # frozen monitoring reference (promote does this too)
uv run python scripts/simulate_feedback.py --fresh                             # delayed labels played forward, ~1 min
uv run python scripts/retrain.py --label-maturity-days 30 --out-dir models/retrain_delay30 --report-dir reports/retrain/delay30
uv run python scripts/ablation.py --model-config configs/model/xgboost_v2_tuned.yaml
uv run python scripts/subgroups.py                                             # champion by subgroup, validation
uv run python scripts/evaluate_test.py --run-name xgb_f5_capacity              # reporting window; logged in ADR 0002
uv run python scripts/final_report.py --run-name xgb_f5_capacity               # reports/final/ + error table
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db                      # browse runs
```

Add `--dev` to `train.py` / `tune.py` for the 10 % sample (seconds, for
plumbing only). Any model config can be trained the same way; each run
writes `models/<run_name>.joblib` and an MLflow run.

**Serve:**

```bash
uv run uvicorn fraud.serve.app:app --port 8000
# http://127.0.0.1:8000/        analyst dashboard: upload a CSV, get a ranked review queue
# http://127.0.0.1:8000/single  one transaction as JSON
# http://127.0.0.1:8000/docs    OpenAPI
curl http://127.0.0.1:8000/health      # {"status":"ok","model_version":"…","parity_rows":50,"audit_events":201,"auth":"open","admin":"disabled","build_commit":null}
curl -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \
  -d '{"TransactionID":3000001,"TransactionDT":12000000,"TransactionAmt":49.0,"ProductCD":"W",
       "card1":9500,"card4":"visa","card6":"debit","C1":1,"D1":120}'
```

Response:

```json
{"transaction_id":3000001,"fraud_probability":0.14656427775136324,"risk_level":"medium","model_version":"xgb_f5_capacity+sigmoid@7af85ec92813"}
```

**No `action` here, deliberately.** `risk_level` is the band the probability
falls in — `high` is at or above the block threshold (`/model-info` →
`bands`). The served policy blocks by threshold and then reviews the
highest-scoring remainder up to the day's analyst budget
(`docs/review_policy.md`), and that second half cannot be decided for one
transaction: whether a score belongs in the day's top *N* depends on the
day's other scores, which a single request does not have. Returning
approve/review here would be fixed bands — a different policy than the
experiments chose. `/predict/batch` and `/predict/csv` have the day in hand
and do return one `action` per row. The evaluation threshold 0.08 (ADR 0006)
is reporting material and is not served — the API never returns two
competing decisions.

Any of the 393 transaction and 40 identity columns may be sent; unknown
fields are rejected; `TransactionID`, `TransactionDT`, `TransactionAmt`,
`ProductCD` and `card1` are required. Omitting every identity field means
"no identity record".

**Batch** — up to 1,000 transactions in one call, returned ranked by fraud
probability. The actions follow the review policy the experiments
recommended (`docs/review_policy.md`): **block** by probability threshold,
**review** the highest-scored remaining transactions up to the analyst
budget, **approve** the rest — so the review volume holds when the score
distribution drifts (a fixed threshold reviewed 141 on a validation day and
222 on a test day; the rank policy reviews exactly the budget on both).
The budget is **per transaction day**, released through the day and
shared by every request that scores that day: the audit trail remembers
what is already under review, a re-scored transaction is re-decided rather
than counted twice, and the response reports the capacity this request had
left (`policy.review_capacity`, `policy.budget_accounting`). One
validation day as ten batches reviews exactly 200 in total, spread across
them.

This cross-request accounting depends on the local audit database. A restart
or redeploy on Render's ephemeral disk loses the budget history. Replicas with
separate SQLite databases would track separate budgets. The current deployment
is a single-instance free demo; production scaling requires shared durable
policy state with atomic budget updates.

```bash
curl -X POST http://127.0.0.1:8000/predict/batch -H 'content-type: application/json' \
  -d '{"review_budget": 200, "transactions":[{...},{...},{...}]}'
# {"n":3,"ranked":[{"rank":1,"transaction_id":…,"fraud_probability":0.1466,"risk_level":"medium","action":"review",…},…],
#  "counts":{"approve":0,"review":3,"block":0},
#  "policy":{"policy":"rank","block_threshold":0.42,"review_budget":200,"review_threshold":null,"review_cutoff":0.1007}}
```

`review_budget` defaults to the server's `default_review_budget` (200);
`"policy": "threshold"` selects the fixed-band policy instead. The same
options apply to `POST /predict/csv?review_budget=200`.

Single `/predict` is **scoring-only**: it returns the probability and the risk
band, not an action. Blocking is a threshold and could be decided for one
transaction, but whether a score is among the day's highest remaining depends on
the day's other scores, which one request does not have — so returning
approve/review there would apply fixed bands, a different policy from the one the
experiments chose. `risk_level: "high"` means at or above the block threshold;
review selection belongs to the queue (`docs/review_policy.md`).

**Analyst dashboard** (`/`): set the analyst review capacity, upload a CSV
of transactions with the IEEE-CIS column names (up to 5,000 rows and 25 MB
— both enforced before the body is parsed or held; extra columns such as a
label are ignored and reported). The server scores it in
one pass through the same input path the parity check uses
(`POST /predict/csv`) and applies the rank policy; the page shows analysed /
block / review / approve / average probability (with the lowest probability
actually reviewed), a table ranked by fraud probability with sort, filter
(flagged / high / review / low) and transaction-id search, a row inspector
showing every non-empty field, and a download of the ranked queue. Changing
the capacity re-applies the policy to the same file. "Try the sample" loads a **synthetic** 200-row
CSV (no Kaggle rows) with a few planted archetype transactions and sets the
capacity to 25 so all three actions appear.

**Audit trail** — every scored transaction is persisted (SQLite, standard
library, one row per transaction): what was scored, when (UTC), by which
model version, under which policy (block threshold, review budget or
threshold, review cutoff), the probability, the action, the request id
(from `X-Request-ID` or generated and echoed back), batch size and request
latency. `GET /audit/recent?limit=50&transaction_id=…` reads the tail;
`/health` reports the row count. Path: `audit_db` in `configs/serving.yaml`,
overridden by `FRAUD_AUDIT_DB`; `null` disables it. In Docker the file lives
on the `/app/audit` volume.

`/audit/recent`, `/audit/monitor` and `POST /outcomes` are admin endpoints:
they answer only to `FRAUD_ADMIN_API_KEY` and return 403 while it is unset
(the public demo).

```bash
export FRAUD_ADMIN_API_KEY=dev     # before starting the server
curl -H "X-API-Key: $FRAUD_ADMIN_API_KEY" "http://127.0.0.1:8000/audit/recent?limit=1"
# [{"id":201,"request_id":"1057101e…","endpoint":"/predict/csv","scored_at":"2026-09-15T16:27:43.597+00:00",
#   "transaction_id":2987069,"model_version":"xgb_f5_capacity+sigmoid@7af85ec92813","fraud_probability":0.0172,
#   "risk_level":"low","action":"approve","policy":"rank","block_threshold":0.42,"review_threshold":null,
#   "review_budget":25,"review_cutoff":0.179,"batch_size":200,"latency_ms":577.7}]
```

**Monitoring** (`docs/monitoring.md`) — the audit trail also stores every
request (including errors, with latency) and a compact input snapshot per
scored row. `scripts/monitor_reference.py` freezes what "normal" looked like
(training-window features; validation-window scores, actions and
performance) and `scripts/monitor.py --since … --until … [--labels …]`
reports API rate / latency / errors, score and action drift (PSI),
per-feature data drift, and — with matured labels — eventual PR-AUC, block
precision, recall and calibration against the reference. Demonstrated on
one validation day (eventual PR-AUC 0.655) and one reporting-window day
(0.617, Brier worse) in `reports/monitoring/`.
A scheduled GitHub Action (`.github/workflows/monitor.yml`, ADR 0011) closes
the loop to the point of notification: daily it asks the live service for the
same report over `GET /audit/monitor` — the service owns the audit trail, so
it runs `build_report` itself and only aggregates leave it — keeps the report
as a run artifact, and opens one labelled issue per episode when the status is
`alert`, commenting while it lasts and closing it on the first healthy run. A
window with fewer than 200 predictions is recorded as `no_data` and never
alerts. Nothing retrains or promotes automatically; that decision stays with
the operator (ADR 0010, `docs/retraining.md`). What the free demo can report is
bounded by the free demo: its instance sleeps and its SQLite audit trail does
not survive a restart, so the job demonstrates the mechanism, not a month of
uninterrupted production monitoring.

```bash
FRAUD_ADMIN_API_KEY=… uv run python scripts/monitor.py \
    --from-url https://fraud-risk-scoring-m1fp.onrender.com --window-hours 24
uv run python scripts/alert.py reports/monitoring/<stamp>.json --source <url> --dry-run
```

**Retraining** (`docs/retraining.md`, ADR 0012) — the monthly lifecycle is
replayed offline by `scripts/retrain.py` (that is where the staleness numbers
come from), and run for real, one cut-off at a time, by
`scripts/retrain_cycle.py` + `.github/workflows/retrain.yml`:

```
label feed clock (+ the monitor's eventual PR-AUC)
  → is a cycle due?          a month of matured labels the champion has not seen,
                             or measured degradation with new data to answer it
  → fit the challenger       on days ≤ train_end, calibrated out-of-fold inside it
  → score both models        on train_end+1 … mature_through, which neither was fit on
                             (the champion is re-scored there, never quoted from its
                              manifest — those numbers belong to another window)
  → gates + margin           ADR 0010's gates, plus "better by 0.005", which they do
                             not require and an unattended job should
  → promote · publish        models/champion/, registry alias, champion-<sha12> release
  → pull request             configs/serving.yaml: the new bands + champion_sha256
  → merge → deploy           deploy-champion.yml points the host at it and verifies
```

One step is not automated, on purpose: a human reads that diff. The block and
review bands decide what happens to a customer's transaction, and the service
takes them from a reviewed commit rather than from a fetched manifest
(`docs/security.md`). Nothing retrains, promotes or deploys behind that gate.

What this dataset lets the automation *do* is narrow, and the pipeline says so
rather than pretending otherwise: under the host's 120-day label maturity, 183
days of data leave no evaluation month the current champion was not itself
trained on, so a scheduled cycle stops and explains which rule stopped it.

**Promotion** (`docs/promotion.md`, ADR 0010) — the service loads
`models/champion/`, which only `scripts/promote.py` writes: a candidate is
measured on validation under its own re-derived policy, must clear every
gate against the current champion (PR-AUC within 0.005, recall at 200
reviews/day, ECE ≤ 0.01, Brier, block precision, cost per transaction) and
reproduce its golden; every evaluation is a registry version with the
verdict in its tags, and the `champion` alias moves only on success. E022
is champion (v1); E016 as a dry-run candidate is rejected on three gates.

**Delayed labels** (`docs/feedback.md`, ADR 0009) — outcomes arrive after
the fact: `POST /outcomes` (or `scripts/feedback.py`, which simulates the
chargeback stream from the dataset's labels) stores them next to the
predictions, and the monitor ages every scored transaction against the
host's 120-day reporting window. Eventual metrics come from closed cohorts
only; the arrived-label positive rate is shown beside the true one because
for 90 days after a month ends it is 100 %. `scripts/simulate_feedback.py`
plays the validation month forward (`reports/feedback/timeline.csv`);
`scripts/retrain.py --label-maturity-days D` runs the lifecycle on labels
that have actually matured and scores the month each artifact served.

**Explain** (`docs/explanation.md`) — `POST /explain` with a `/predict`
body returns the top-*k* signed contributions (XGBoost `pred_contribs`,
summed per source column and per feature family, with the row's values),
the bias, the raw score they sum to and the calibrated probability the
service returns. The dashboard's row inspector shows it as *Why this
score*. It explains why a transaction ranks where it does, not whether the
purchase was fraud.

**Access** — two keys, two audiences. `FRAUD_API_KEY` guards scoring and
explanation (`/predict*`, `/explain`); unset, they are open, `/health`
reports `auth: open`, and the dashboard needs no key. `FRAUD_ADMIN_API_KEY`
guards the operational endpoints (`POST /outcomes` writes the delayed-label
store, `GET /audit/recent` reads scored rows, `GET /audit/monitor` builds the
monitoring report over them); unset, they return 403 and
`/health` reports `admin: disabled`. Neither key unlocks the other's routes.
The public demo sets neither: anyone can score, nobody can write labels or
read the audit trail. Every guarded call is logged as one JSON line with its
request id and bounded by `request_timeout_s` (`docs/deployment.md` →
Hardening).

**Rate limits** — per client in any 60 seconds: 60 single predictions,
5 CSV/JSON batch requests combined, and 10 explanations. Excess calls return
`429` with `Retry-After` before body parsing or inference. Configured in
`configs/serving.yaml`; counters live in one process and reset on restart.
See [deployment hardening](docs/deployment.md#hardening-for-a-public-url) for
Render client identification and local benchmark configuration.

**Model info** — what is being served:

```bash
curl http://127.0.0.1:8000/model-info
# {"model":"xgboost","default_policy":"rank","default_review_budget":200,"experiment":"E022","version":"xgb_f5_capacity+sigmoid@7af85ec92813",
#  "feature_set":"f5_interactions","n_inputs":439,"primary_metric":"pr_auc","validation_pr_auc":0.6367,"test_pr_auc":0.561,
#   (test_pr_auc is null for a champion retrained through the test window — ADR 0012)
#  "calibration":"sigmoid","bands":{"review":0.062,"block":0.42},
#  "training_window_days":[1,122],"validation_window_days":[123,152],"parity_rows":50}
```

**Docker:**

```bash
docker build -t fraud-risk-scoring .            # needs the calibrated artifact + its frozen sample
docker run --rm -p 8000:8000 -v fraud-audit:/app/audit fraud-risk-scoring   # volume keeps the audit trail
```

**Public deployment** (`docs/deployment.md`): a Render free web service
(512 MB, 0.1 CPU, no card) built from `deploy/hosted/Dockerfile`, which
holds no model — at startup the service fetches `models/champion/` from
the `champion-<sha>` pre-release of this repository (the published
weights, pinned by content hash; `scripts/publish_champion.py`), then
runs the same parity check as everywhere else. Measured under those limits: 238 MiB at
rest, cold start ~1 min, the 200-row sample in ~19 s. Fly.io remains wired
as the paid alternative (`fly.toml`). `scripts/deploy_check.py <url>`
verifies health + parity, model-info and a scored sample on any deployment.

**Production release:** `uv run python scripts/release.py vX.Y.Z --full-checks` runs
the fixture and real-data tests (no skipped slow tests allowed), lint and type checks,
verifies the champion and pushes the tag; `.github/workflows/deploy.yml`
then re-runs CI, triggers the Render deploy of the tag (and the Fly image
if configured), waits for the tag's version to serve, runs
`deploy_check.py` against the live URL and cuts the GitHub release. No
runner ever sees the weights.

The full preflight requires the IEEE training CSVs, champion files and source
training pipeline on the release machine; see [deployment](docs/deployment.md#release-and-continuous-deployment).

## Performance

`uv run python scripts/benchmark_api.py` starts a local server on a scratch
audit database and measures it; `--url` benchmarks a running one. It uses
the 200 synthetic rows shipped with the dashboard, tiled to the API limits,
so it runs from a clean clone. Numbers below are from
`reports/benchmark/`: the Docker image under `--cpus 1 --memory 1g` (the
`fly.toml` sizing) on an Apple M-series host, one uvicorn worker.

| scenario | p50 | p95 / max |
|---|---:|---:|
| container cold start → healthy `/health` (model load + parity check) | 3.5 s | |
| single `/predict`, sequential | 46 ms | 77 ms |
| `/predict/batch`, 1,000 rows | 0.59 s | 0.62 s |
| `/predict/csv`, 200 rows (the dashboard sample) | 0.51 s | 0.54 s |
| `/predict/csv`, 1,000 rows | 1.4 s | 1.5 s |
| `/predict/csv`, 5,000 rows (4.7 MB, the upload limit) | 6.4 s | 6.6 s |
| 20 concurrent clients, single `/predict` | 0.93 s | 1.19 s — **21 req/s**, 0 errors |
| 20 concurrent 200-row CSV uploads | 6.6 s | 10.5 s — 381 rows/s |
| peak container memory (during the concurrent CSV run) | **538 MiB** | |

What this says about `fly.toml`: 1 GB leaves ~2× headroom over the peak;
the 60 s health-check grace period covers a 3.5 s cold start twenty times
over; the soft concurrency limit of 20 is where single-prediction latency
has already risen from 46 ms to ~1 s, because one worker serialises
scoring — so 20 is the right soft limit for one machine, and throughput
beyond ~20 req/s means more machines, not a bigger one. The 8-CPU host
without limits gives the same numbers (`reports/benchmark/local.md`):
scoring is single-threaded at these batch sizes.

The benchmark found one bug on its first run: `/predict/batch` with 1,000
rows took 36 s (a one-row DataFrame per transaction, built twice); it is
now 0.5 s.

## Data sources & schema

Kaggle *IEEE-CIS Fraud Detection* (Vesta Corporation), labelled training
tables only: `train_transaction.csv` (590,540 × 394) and
`train_identity.csv` (144,233 × 41), joined on `TransactionID`. Kaggle's
unlabelled test files are downloaded for completeness and not used. The
exact column contract — names, string vs numeric, non-null set — is
`src/fraud/data/schema.py`, checked at load time and by `tests/test_data.py`.
Column families and what was inferred about them: `docs/eda.md`.

## Politeness & legal

- Data is fetched only through Kaggle's official API with the user's own
  token, after accepting the competition rules; nothing is scraped and no
  rate-limited endpoint is polled.
- **Not committed:** the dataset, any row of it, the Parquet cache, trained
  artifacts, MLflow runs, or tokens. Test fixtures are synthetic rows
  generated by `tests/fixtures/make_fixtures.py`.
- The data is anonymised by the provider, and this project collects and stores
  no personal data of its own. **The running service is a different matter:** it
  writes what callers submit to a prediction audit trail — transaction id, amount,
  product, card network and type, device type and derived flags — because
  demonstrating drift monitoring needs a record of what was scored. On the public
  demo that store is ephemeral and world-writable by anyone who can reach
  `/predict`, which is why the page says to submit synthetic data only. Do not put
  real cardholder or customer records into it.
- **Demographic fairness cannot be assessed**, because the
  dataset contains no protected demographic attributes, so no fairness claim is
  made. **Operational subgroup robustness has been evaluated** — product, card
  network and type, identity availability, e-mail provider family, device class,
  amount band and ten-day time block: no group is scored above its own fraud
  rate, and the weak segment (no identity record) is a coverage gap rather than
  an over-flagging one. See the [model card](docs/model_card.md) and
  [subgroup analysis](docs/subgroups.md).

## Security

The service loads a pickle it fetches over the network, so the risk here is
mostly supply chain. `model.joblib` is verified against the digest in its
release URL *before* it is deserialized; the deploy is pinned to the tagged
commit; and a release fails unless the live service reports both that commit
and the champion the tag was cut for.

Dependencies are watched rather than assumed: Dependabot opens weekly PRs for
`uv`, GitHub Actions and both Dockerfiles, and a weekly `security.yml` runs
`pip-audit` over the locked runtime and development sets plus CodeQL. Weekly
matters — a CVE against an untouched dependency still needs to surface.

What is *not* covered (no SBOM, no image scanning, no artifact signing, tags
rather than SHAs for actions) is listed explicitly in
[docs/security.md](docs/security.md).

## Data storage

`data/raw/` (CSVs), `data/processed/train.parquet` (81 MB cache, safe to
delete), `models/` (joblib artifacts, frozen goldens, the local audit
trail `models/audit/prediction_events.sqlite`), `mlflow.db` + `mlruns/`
(experiment ledger), `reports/` (committed evidence). Everything except
`reports/` is git-ignored.

## Testing

```bash
uv run pytest              # 200 fixture tests, no data, no network, ~20 s
uv run pytest -m slow      # 4 tests against the real files and the production artifact
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Tests are grouped by what they protect: data contract, split ordering and
sizes, leakage (nothing fit on validation; history features depend on no
later row), pipeline (unseen categories, missing identity, feature count,
save/load parity), model (beats the prior, probabilities in [0, 1], shape,
reproducible), threshold/calibration arithmetic, serving (validation, API =
offline, batch = single, CSV = single, OpenAPI contract), parity (training
path = artifact = API = frozen golden; tampering refused). The property →
test map is `docs/testing.md`.

CI (`.github/workflows/ci.yml`): ruff → mypy → pytest → smoke training
through the CLI on the fixture → Docker build → container `/health` with
the startup parity check. The full model is never trained in CI. A version
tag additionally deploys the release image and verifies the live service
(`.github/workflows/deploy.yml`, `docs/deployment.md` → Release).

## Development setup

```bash
uv sync                                 # includes the train and dev groups
uv run python tests/fixtures/make_fixtures.py   # regenerate the synthetic fixture
```

Branch per phase, PR per feature, CI on every PR (lint, format, types,
tests, Docker build against a fixture-trained stand-in artifact). New
features enter through `docs/leakage_audit.md` and an experiment entry with
the hypothesis written before the run.

## Limitations

**Autonomous blocking is a demonstration policy.** In a real payment system,
weak subgroups would need additional policy review and likely more conservative
treatment. The overall block-precision target does not hold in every subgroup;
see the [model card](docs/model_card.md) and [subgroup analysis](docs/subgroups.md).

See `docs/model_card.md`. In short: the label is partly account-level; the
provider's engineered columns are taken as point-in-time on the host's
word; one month of drift costs 0.08 PR-AUC; the cost model is assumed and
the threshold moves with it; feature computation is stateless, but prediction
auditing and cross-request review-budget accounting are stateful; production
labels mature 120 days after the transaction, so a month's eventual
performance is unknown for four months (`docs/feedback.md`); the global
PR-AUC blends 0.80 on transactions with an identity record and 0.46 on the
82 % without one (`docs/subgroups.md`).

## License

The original project code and documentation are available under the
[MIT License](LICENSE). This does not license the IEEE-CIS dataset or other
third-party materials; their respective terms and licenses still apply.
