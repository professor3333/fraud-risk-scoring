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
  metrics is the object the API loads. A test asserts API scores equal the
  offline object's on the same raw rows.
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
for every decision. Test window = days 153 – 183 (85,430 / 2,994), scored
once at the end.

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

Final model (E016 + sigmoid calibration, threshold 0.08):

| metric | validation | test |
|---|---:|---:|
| PR-AUC | 0.619 | **0.557** |
| ROC-AUC | 0.932 | 0.910 |
| precision / recall / F1 at 0.08 | 0.335 / 0.730 / 0.459 | 0.275 / 0.717 / 0.397 |
| recall reviewing the top 500 transactions per day | 0.860 | 0.816 |
| Brier (prior 0.033) / ECE | 0.0191 / 0.0049 | 0.0217 / 0.0067 |
| cost at 0.08 vs approve-all | 230k vs 484k (−52 %) | 262k vs 478k (−45 %) |

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
- Training CLI logging params, split, feature list, full metric set,
  PR curve, confusion matrix and the fitted pipeline to MLflow.
- Expanding-window random search; post-hoc learning curves.
- Sigmoid / isotonic calibration on out-of-fold scores, chosen by rule.
- Amount-weighted cost curve, sensitivity table, train-only cross-check.
- Gain, group-permutation and group-ablation importance.
- Single test-window evaluation with top-*k*-per-day review metrics.
- FastAPI service (`/health`, `/predict`, demo page at `/`), Dockerfile.
- 63 fixture-based tests (no data, no network) + 3 slow real-data tests.

## Tech stack

Python 3.12 · uv · pandas 3 · scikit-learn · XGBoost 3 · MLflow 3 (SQLite)
· FastAPI + Pydantic + uvicorn · matplotlib · pytest · ruff · mypy · Docker.

## Project structure

```
configs/          split.yaml, dev.yaml, features/*.yaml, model/*.yaml,
                  tuning/xgboost.yaml, threshold.yaml, serving.yaml
data/             git-ignored; data/README.md explains the download
docs/             eda.md, decisions/ (ADR 0001–0007), experiments.md,
                  leakage_audit.md, threshold.md, ablation.md, feature_sets.md,
                  model_card.md
reports/          committed evidence: EDA figures, curves, calibration,
                  threshold, ablation, test
scripts/          download_data, validate_data, eda, train, tune, learning_curve, calibrate,
                  select_threshold, ablation, feature_ladder, evaluate_test,
                  split_comparison, make_fixture_artifact
src/fraud/
  data/           schema (contract), validate (checks), load (read + join), split
  features/       columns (spec), derive, time, rowwise (F1/F2/F4/F5), encoders, history
  pipeline/       build (preprocessing + model), calibrated
  train/          run (MLflow), tune (expanding-window CV)
  evaluate/       metrics, curves, calibration, threshold, importance
  serve/          app, schemas, static/index.html
tests/            fixtures/ (synthetic 400-row raw files + generator), test_*.py
Dockerfile        runtime-only image, non-root
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
uv run pytest            # 72 tests on the synthetic fixture; no data needed
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
uv run python scripts/train.py --model configs/model/xgboost_f5_interactions.yaml  # E016, shipped
uv run python scripts/learning_curve.py --model models/xgb_v2_tuned.joblib
uv run python scripts/calibrate.py --model-config configs/model/xgboost_f5_interactions.yaml
uv run python scripts/select_threshold.py --run-name xgb_f5_interactions
uv run python scripts/ablation.py --model-config configs/model/xgboost_v2_tuned.yaml
uv run python scripts/evaluate_test.py --run-name xgb_f5_interactions          # once per candidate
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db                      # browse runs
```

Add `--dev` to `train.py` / `tune.py` for the 10 % sample (seconds, for
plumbing only). Any model config can be trained the same way; each run
writes `models/<run_name>.joblib` and an MLflow run.

**Serve:**

```bash
uv run uvicorn fraud.serve.app:app --port 8000
# then open http://127.0.0.1:8000/  (demo page)  ·  http://127.0.0.1:8000/docs  (OpenAPI)
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \
  -d '{"TransactionID":3000001,"TransactionDT":12000000,"TransactionAmt":49.0,"ProductCD":"W",
       "card1":9500,"card4":"visa","card6":"debit","C1":1,"D1":120}'
```

Response:

```json
{"transaction_id":3000001,"fraud_probability":0.072,"decision":"approve","threshold":0.08,"model_version":"xgb_f5_interactions+sigmoid@9c33a1cdc4db"}
```

Any of the 393 transaction and 40 identity columns may be sent; unknown
fields are rejected; `TransactionID`, `TransactionDT`, `TransactionAmt`,
`ProductCD` and `card1` are required. Omitting every identity field means
"no identity record".

**Docker:**

```bash
docker build -t fraud-risk-scoring .            # needs models/xgb_f5_interactions_calibrated.joblib
docker run --rm -p 8000:8000 fraud-risk-scoring
```

The service is **local-only** in this build: the artifact is trained from
competition data whose rules restrict redistribution, and a public endpoint
scoring that model would publish it. The image runs anywhere Docker does.

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
delete), `models/` (joblib artifacts), `mlflow.db` + `mlruns/` (experiment
ledger), `reports/` (committed evidence). Everything except `reports/` is
git-ignored.

## Testing

```bash
uv run pytest              # 72 fixture tests, no data, no network, ~10 s
uv run pytest -m slow      # 3 tests against the real files, if present
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Tests are grouped by what they protect: data contract, split ordering and
sizes, leakage (nothing fit on validation; history features depend on no
later row), pipeline (unseen categories, missing identity, save/load
parity), model (beats the prior, probabilities in [0, 1], reproducible),
threshold/calibration arithmetic, serving (validation, API = offline).

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
