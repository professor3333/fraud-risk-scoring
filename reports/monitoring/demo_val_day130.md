# Monitoring report — ALERT

Generated 2026-09-15T22:49:04+05:45; window 2026-09-15T16:51:47.113+00:00 → 2026-09-15T16:52:30.753+00:00.

## Flags

- feature drift alert: n_missing_transaction (PSI 1.094)
- error rate 16.7%

## API

- requests 12 (0.294/s over 41 s), rows scored 2807
- latency p50 3535 ms · p95 4800 ms · max 5058 ms
- errors 2 (16.7%)

| endpoint | requests | rows | p50 ms | p95 ms | errors |
|---|---:|---:|---:|---:|---:|
| `/predict` | 2 | 0 | 9 | 14 | 2 |
| `/predict/csv` | 10 | 2807 | 3814 | 4847 | 0 |

## Predictions

- rows 2807 · mean probability 0.0303 (reference 0.0358) · p50 0.0065 · p99 0.8121
- score PSI 0.021 → **ok**
- actions: block 1.7% · review 7.1% · approve 91.2% (reference 2.0% / 7.1% / 91.0%); PSI 0.001 → **ok**
- model versions: {'xgb_f5_capacity+sigmoid@7af85ec92813': 2807}; policies: {'rank': 2807}

## Data

| feature | PSI | drift | now vs reference |
|---|---:|---|---|
| product | 0.084 | ok | W 83.3%, C 8.7%, H 3.3% |
| card4 | 0.019 | ok | visa 65.8%, mastercard 32.3%, american express 1.0% |
| card6 | 0.031 | ok | debit 78.9%, credit 21.1%, debit or credit 0.0% |
| device_type | 0.057 | ok | <missing> 83.8%, desktop 9.7%, mobile 6.6% |
| hour | 0.058 | ok | 21 7.5%, 15 7.3%, 20 7.1% |
| has_identity | 0.060 | ok | rate 0.165 vs 0.265 |
| addr1_missing | 0.009 | ok | rate 0.086 vs 0.115 |
| p_email_present | 0.002 | ok | rate 0.829 vs 0.845 |
| r_email_present | 0.051 | ok | rate 0.159 vs 0.248 |
| amount | 0.014 | ok | mean 136.89 vs 134.48 |
| n_missing_transaction | 1.094 | alert | mean 192.42 vs 161.25 |
| n_missing_identity | 0.060 | ok | mean 35.80 vs 33.25 |

## Model

- reference (validation): PR-AUC 0.637 · ROC-AUC 0.933 · block precision 0.80 · Brier 0.0185 · ECE 0.0037
- eventual (2807 labelled, positive rate 0.034): PR-AUC 0.655 (+0.018) · ROC-AUC 0.882
- block precision 0.89 (+0.09) · recall block 0.44 · recall block+review 0.78
- calibration: Brier 0.0178 · ECE 0.0110
