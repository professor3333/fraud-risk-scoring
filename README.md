# Fraud Risk Scoring

[![ci](https://github.com/professor3333/fraud-risk-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/professor3333/fraud-risk-scoring/actions/workflows/ci.yml)

A fraud risk scoring system built on the Kaggle IEEE-CIS Fraud Detection
dataset. Given one online transaction, it returns a calibrated probability
that the transaction is fraudulent and a decision at an operating threshold
chosen from a written cost model — served over a FastAPI endpoint, with every
modelling decision recorded and every improvement measured on a strictly
later window of time.

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
   └────────────────────┬────────────────────┘             decision at 0.08,
                        │                                   model version
        CalibratedModel (sigmoid map fit on
        out-of-fold training-window scores)
                        │
   MLflow: params, split, features, metrics, curves, artifact
   threshold: amount-weighted cost curve on validation → 0.08
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
| E007 | + card-history features, earlier rows only (ADR 0004) | 0.582 · rejected (−0.002 seed-paired) | 0.920 | 0.280 |
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
validation became +0.004 on the test month and its operating-point numbers
moved slightly the other way: more capacity memorises the training window
harder and transfers a little worse. It shipped because validation decides
and test reports (`docs/experiments.md`); the open methodology item is a
drift-aware validation horizon.

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
can compute. Ablation shows the 339 `V`
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
- Expanding-window random search; post-hoc learning curves.
- Sigmoid / isotonic calibration on out-of-fold scores, chosen by rule.
- Amount-weighted cost curve, sensitivity table, train-only cross-check;
  three-action review policy sized to an analyst budget.
- Gain, group-permutation and group-ablation importance.
- Single test-window evaluation with top-*k*-per-day review metrics.
- FastAPI service: `/health`, `/predict`, `/predict/batch` and `/predict/csv`
  returning one authoritative `action` (approve / review / block) from the
  rank-based review policy, `/model-info`, `/audit/recent`; startup parity
  check against a frozen golden; every scored transaction persisted to a
  SQLite audit trail with model version, policy, action and latency.
- Analyst dashboard at `/`: CSV upload → summary tiles, ranked table with
  sort / filter / search, row inspector, ranked-CSV download; single-
  transaction form at `/single`. Dockerfile.
- 63 fixture-based tests (no data, no network) + 3 slow real-data tests.

## Tech stack

Python 3.12 · uv · pandas 3 · scikit-learn · XGBoost 3 · MLflow 3 (SQLite)
· FastAPI + Pydantic + uvicorn · a dependency-free HTML/JS dashboard ·
matplotlib · pytest · ruff · mypy · Docker.

## Project structure

```
configs/          split.yaml, dev.yaml, features/*.yaml, model/*.yaml,
                  tuning/*.yaml, threshold.yaml, policy.yaml, serving.yaml
data/             git-ignored; data/README.md explains the download
docs/             eda.md, decisions/ (ADR 0001–0007), EXPERIMENT_LOG.md (4-column
                  ledger), experiments.md (long form), leakage_audit.md,
                  threshold.md, review_policy.md, ablation.md, feature_sets.md,
                  error_analysis.md, xgboost_progression.md, testing.md,
                  deployment.md, defending_the_decisions.md, model_card.md
reports/          committed evidence: EDA figures, curves, calibration,
                  threshold, policy, ablation, feature sets, test, final
scripts/          download_data, validate_data, eda, train, tune, param_sweep,
                  learning_curve, calibrate, select_threshold, review_policy,
                  freeze_artifact, ablation, feature_ladder, evaluate_test,
                  final_report, split_comparison, deploy_check,
                  make_fixture_artifact
src/fraud/
  data/           schema (contract), validate (checks), load (read + join), split
  features/       columns (spec), derive, time, rowwise (F1/F2/F4/F5), encoders, history
  pipeline/       build (preprocessing + model), calibrated
  train/          run (MLflow), tune (expanding-window CV)
  evaluate/       metrics, curves, calibration, threshold, importance
  serve/          app, schemas, frames, parity (frozen-golden check), audit
                  (SQLite prediction events), static/ (dashboard, single form,
                  synthetic sample CSV)
tests/            fixtures/ (synthetic 400-row raw files + generator), test_*.py
Dockerfile        runtime-only image, non-root
fly.toml          Fly.io app definition (public deployment)
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
uv run pytest            # 99 tests on the synthetic fixture; no data needed
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
uv run python scripts/learning_curve.py --model models/xgb_v2_tuned.joblib
uv run python scripts/calibrate.py --model-config configs/model/xgboost_f5_capacity.yaml
uv run python scripts/select_threshold.py --run-name xgb_f5_capacity
uv run python scripts/review_policy.py --run-name xgb_f5_capacity --with-test
uv run python scripts/freeze_artifact.py --run-name xgb_f5_capacity            # frozen golden for parity
uv run python scripts/ablation.py --model-config configs/model/xgboost_v2_tuned.yaml
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
curl http://127.0.0.1:8000/health      # {"status":"ok","model_version":"…","parity_rows":50,"audit_events":201}
curl -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \
  -d '{"TransactionID":3000001,"TransactionDT":12000000,"TransactionAmt":49.0,"ProductCD":"W",
       "card1":9500,"card4":"visa","card6":"debit","C1":1,"D1":120}'
```

Response:

```json
{"transaction_id":3000001,"fraud_probability":0.1466,"risk_level":"medium","action":"review","model_version":"xgb_f5_capacity+sigmoid@7af85ec92813"}
```

`action` is the one command for the payment system (approve / review /
block) and `risk_level` its low / medium / high reading, from the review
policy (`docs/review_policy.md`). The evaluation threshold 0.08 (ADR 0006)
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

```bash
curl -X POST http://127.0.0.1:8000/predict/batch -H 'content-type: application/json' \
  -d '{"review_budget": 200, "transactions":[{...},{...},{...}]}'
# {"n":3,"ranked":[{"rank":1,"transaction_id":…,"fraud_probability":0.1466,"risk_level":"medium","action":"review",…},…],
#  "counts":{"approve":0,"review":3,"block":0},
#  "policy":{"policy":"rank","block_threshold":0.42,"review_budget":200,"review_threshold":null,"review_cutoff":0.1007}}
```

`review_budget` defaults to the server's `default_review_budget` (200);
`"policy": "threshold"` selects the fixed-band policy instead. The same
options apply to `POST /predict/csv?review_budget=200`. Single `/predict`
has no batch to rank within, so it reports the fixed bands.

**Analyst dashboard** (`/`): set the analyst review capacity, upload a CSV
of transactions with the IEEE-CIS column names (up to 5,000 rows; extra
columns such as a label are ignored and reported). The server scores it in
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

```bash
curl "http://127.0.0.1:8000/audit/recent?limit=1"
# [{"id":201,"request_id":"1057101e…","endpoint":"/predict/csv","scored_at":"2026-09-15T16:27:43.597+00:00",
#   "transaction_id":2987069,"model_version":"xgb_f5_capacity+sigmoid@7af85ec92813","fraud_probability":0.0172,
#   "risk_level":"low","action":"approve","policy":"rank","block_threshold":0.42,"review_threshold":null,
#   "review_budget":25,"review_cutoff":0.179,"batch_size":200,"latency_ms":577.7}]
```

**Model info** — what is being served:

```bash
curl http://127.0.0.1:8000/model-info
# {"model":"xgboost","default_policy":"rank","default_review_budget":200,"experiment":"E022","version":"xgb_f5_capacity+sigmoid@7af85ec92813",
#  "feature_set":"f5_interactions","n_inputs":439,"primary_metric":"pr_auc","validation_pr_auc":0.6367,"test_pr_auc":0.561,
#  "calibration":"sigmoid","bands":{"review":0.062,"block":0.42},
#  "training_window_days":[1,122],"validation_window_days":[123,152],"parity_rows":50}
```

**Docker:**

```bash
docker build -t fraud-risk-scoring .            # needs the calibrated artifact + its frozen sample
docker run --rm -p 8000:8000 -v fraud-audit:/app/audit fraud-risk-scoring   # volume keeps the audit trail
```

**Public deployment:** Fly.io, from the same image (`fly.toml`,
`docs/deployment.md`): the image sits in Fly's private registry so the
model weights are not redistributed, the endpoint is public, and
`scripts/deploy_check.py <url>` verifies health + parity, model-info and a
scored sample after each deploy. Requires the account owner's `flyctl auth
login` once.

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
- The data is anonymised by the provider; no personal data is collected or
  stored by this project. The model card notes that fairness across proxies
  such as email domain or device type has not been assessed.

## Data storage

`data/raw/` (CSVs), `data/processed/train.parquet` (81 MB cache, safe to
delete), `models/` (joblib artifacts, frozen goldens, the local audit
trail `models/audit/prediction_events.sqlite`), `mlflow.db` + `mlruns/`
(experiment ledger), `reports/` (committed evidence). Everything except
`reports/` is git-ignored.

## Testing

```bash
uv run pytest              # 99 fixture tests, no data, no network, ~20 s
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
the startup parity check. The full model is never trained in CI.

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

See `docs/model_card.md`. In short: the label is partly account-level; the
provider's engineered columns are taken as point-in-time on the host's
word; one month of drift costs 0.06 PR-AUC; the cost model is assumed and
the threshold moves with it; serving is stateless by design.
