# Operating threshold

Model: `xgb_v2_tuned` (E008) with sigmoid calibration (ADR 0007). Cost model
and procedure: ADR 0006. Evidence: `reports/threshold/`,
`scripts/select_threshold.py`.

## Cost assumptions

| error | cost per transaction of amount *a* |
|---|---|
| approved fraud (FN) | *a* + 15 |
| declined legitimate (FP) | 0.10 · *a* + 2 |

These are assumptions, not measurements; the sensitivity table below is the
honest statement of how much they matter.

## Result on the validation window (85,044 transactions, 2,884 fraud)

| policy | threshold | total cost | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| approve everything | — | 485,742 | — | 0 | — |
| naive 0.5 | 0.50 | 328,071 | 0.838 | 0.404 | 0.545 |
| F1-optimal | 0.28 | 278,523 | 0.692 | 0.511 | 0.588 |
| **cost-optimal (chosen)** | **0.08** | **227,214** | 0.335 | 0.724 | 0.458 |
| cost-optimal on train-only OOF scores | 0.085 | 227,998 | 0.350 | 0.712 | 0.469 |

The chosen operating point cuts the validation month's fraud cost by 53 %
relative to approving everything and by 31 % relative to the naive 0.5. The
train-only cross-check (out-of-fold scores from days 63 – 122, which never
saw validation) lands within one grid step, so the choice is not an artefact
of the validation month.

The policy is recall-heavy: about 7 % of transactions are declined, one in
three of them fraudulent. That follows directly from the assumption that a
false decline costs a tenth of the amount plus $2 while a missed fraud costs
the whole amount. A merchant who values customer friction more would sit
higher on the curve — see the sensitivity table.

## Sensitivity (re-selected threshold under ±50 % on each cost)

| scenario | threshold | total cost | precision | recall |
|---|---:|---:|---:|---:|
| base | 0.080 | 227,214 | 0.335 | 0.724 |
| FN cost −50 % | 0.165 | 139,428 | 0.536 | 0.600 |
| FN cost +50 % | 0.070 | 288,229 | 0.305 | 0.752 |
| FP cost −50 % | 0.055 | 173,860 | 0.254 | 0.789 |
| FP cost +50 % | 0.140 | 259,028 | 0.487 | 0.626 |

The threshold ranges 0.055 – 0.165 across scenarios: the *direction* is
robust (well below 0.5), the exact value is a business input. The curve's
total cost is within 5 % of its minimum for thresholds 0.06 – 0.12, so any
value in that band is defensible under the base assumptions.

## What is fixed from here

`configs/threshold.yaml: threshold: 0.08`. Every later evaluation reports
precision, recall, F1 and the confusion matrix at 0.08 on calibrated
probabilities; the API applies the same value. The test window is evaluated
at this threshold once.
