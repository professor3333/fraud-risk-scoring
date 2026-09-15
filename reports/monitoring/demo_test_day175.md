# Monitoring report — ALERT

Generated 2026-09-15T22:49:10+05:45; window 2026-09-15T16:52:30.760+00:00 → 2026-09-15T16:53:14.001+00:00.

## Flags

- feature drift alert: n_missing_transaction (PSI 0.285)
- error rate 16.7%

## API

- requests 12 (0.297/s over 40 s), rows scored 2683
- latency p50 3683 ms · p95 4574 ms · max 5195 ms
- errors 2 (16.7%)

| endpoint | requests | rows | p50 ms | p95 ms | errors |
|---|---:|---:|---:|---:|---:|
| `/predict` | 2 | 0 | 14 | 24 | 2 |
| `/predict/csv` | 10 | 2683 | 3742 | 4687 | 0 |

## Predictions

- rows 2683 · mean probability 0.0403 (reference 0.0358) · p50 0.0087 · p99 0.7889
- score PSI 0.014 → **ok**
- actions: block 2.1% · review 7.5% · approve 90.4% (reference 2.0% / 7.1% / 91.0%); PSI 0.000 → **ok**
- model versions: {'xgb_f5_capacity+sigmoid@7af85ec92813': 2683}; policies: {'rank': 2683}

## Data

| feature | PSI | drift | now vs reference |
|---|---:|---|---|
| product | 0.055 | ok | W 80.8%, C 9.4%, R 4.4% |
| card4 | 0.017 | ok | visa 64.9%, mastercard 32.6%, discover 1.3% |
| card6 | 0.018 | ok | debit 76.0%, credit 24.0%, <missing> 0.0% |
| device_type | 0.036 | ok | <missing> 82.0%, desktop 10.9%, mobile 7.1% |
| hour | 0.089 | ok | 22 8.0%, 20 7.5%, 21 7.5% |
| has_identity | 0.032 | ok | rate 0.190 vs 0.265 |
| addr1_missing | 0.004 | ok | rate 0.096 vs 0.115 |
| p_email_present | 0.004 | ok | rate 0.820 vs 0.845 |
| r_email_present | 0.022 | ok | rate 0.187 vs 0.248 |
| amount | 0.013 | ok | mean 146.06 vs 134.48 |
| n_missing_transaction | 0.285 | alert | mean 161.33 vs 161.25 |
| n_missing_identity | 0.039 | ok | mean 35.25 vs 33.25 |

## Model

- reference (validation): PR-AUC 0.637 · ROC-AUC 0.933 · block precision 0.80 · Brier 0.0185 · ECE 0.0037
- eventual (2683 labelled, positive rate 0.040): PR-AUC 0.617 (-0.020) · ROC-AUC 0.913
- block precision 0.81 (+0.01) · recall block 0.43 · recall block+review 0.70
- calibration: Brier 0.0228 · ECE 0.0056
