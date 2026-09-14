# Fraud Risk Scoring

End-to-end fraud risk scoring on the Kaggle IEEE-CIS Fraud Detection dataset:
a calibrated fraud probability per transaction plus a decision at an
operationally chosen threshold, served over an API.

Work in progress — see `PROGRESS.md`.

## Setup

```bash
uv sync
uv run python scripts/download_data.py   # needs a Kaggle token; see data/README.md
uv run pytest                            # fixture-based tests, no data needed
```

## Train

```bash
uv run python scripts/train.py --model configs/model/logreg.yaml --dev   # 10 % sample, seconds
uv run python scripts/train.py --model configs/model/xgboost.yaml        # full training window, ~45 s
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db                 # browse runs
```

## Results (validation window, ADR 0002)

| experiment | model | PR-AUC | ROC-AUC | recall @ precision ≥ 0.90 |
|---|---|---:|---:|---:|
| E001 | constant prior | 0.034 | 0.500 | 0.000 |
| E002 | logistic regression, raw columns | 0.402 | 0.842 | 0.118 |
| E003 | XGBoost, raw columns | **0.570** | 0.912 | 0.292 |

Details and interpretation: `docs/experiments.md`. Decisions: `docs/decisions/`.
