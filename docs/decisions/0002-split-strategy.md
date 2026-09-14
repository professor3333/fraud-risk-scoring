# ADR 0002 — Temporal split strategy

**Date:** 2026-09-14 · **Status:** accepted

## Context

- Labelled data covers 182 days (`day = TransactionDT // 86400` ∈ [1, 183]),
  already sorted by time; no day is empty (`docs/eda.md` §2).
- Weeks 0 – 3 (days 1 – 27) are a seasonal regime: product mix, identity
  coverage and fraud rate all differ from the remaining 23 weeks (§5).
- The label propagates within an account (ADR 0001), so any split that puts
  rows of the same account on both sides of a boundary at random leaks.
- Kaggle's own test set is chronologically after the training set with a gap
  of roughly a month; our system is evaluated the same way, on labelled data.

## Options

1. **Random or stratified split.** Rejected: hides the seasonal shift entirely
   and leaks the propagated label through shared entities.
2. **Time split without gaps** — train / validation / test are three
   consecutive windows.
3. **Time split with gaps** — a blank window between train and validation and
   between validation and test, to mimic retraining latency and label
   immaturity.
4. **Exclude weeks 0 – 3 from training** so the model never sees the seasonal
   regime.

## Decision

**Option 2, with the window boundaries below, and option 4 treated as an
experiment rather than a default.**

| window | days (inclusive) | `TransactionDT` range (s) | length | rows | positives | rate |
|---|---|---|---|---:|---:|---:|
| train | 1 – 122 | 86 400 ≤ DT < 10 627 200 | 122 days | 420,066 | 14,785 | 3.52 % |
| validation | 123 – 152 | 10 627 200 ≤ DT < 13 219 200 | 30 days | 85,044 | 2,884 | 3.39 % |
| test | 153 – 183 | 13 219 200 ≤ DT ≤ 15 811 131 | 31 days | 85,430 | 2,994 | 3.50 % |

Rationale:

- **30-day validation and test windows** each contain ≈ 85k transactions
  and ≈ 2.9 – 3.0k positives, enough for stable PR-AUC estimates; a shorter window
  would make the metric noisy, a longer one would starve training.
- **Validation and test start well after the seasonal month** (day 123 vs
  day 27), so both evaluate the regime the model will actually face.
- **No gap.** Features are computed from strictly earlier rows, so a row at
  the boundary is scored with exactly the history a live system would have;
  a gap does not make features more valid. What a gap *would* model is (a)
  retrain latency and (b) training-label immaturity — reports that arrive
  after the training cut-off. Both make our training labels slightly better
  than production's, which biases metrics optimistically by a small,
  time-invariant amount, and does not change model comparisons. Recorded as a
  known limitation in the model card rather than paid for with data.
- **Weeks 0 – 3 stay in training by default.** Removing 118k rows on the
  hypothesis that they hurt is an experiment (`docs/experiments.md`); the
  default is the simplest thing.
- **Time-ordered cross-validation**, if used for tuning, is expanding-window
  inside the training window only (e.g. folds ending at days 60, 90, 122).
  Validation and test are never folded.

## Development subset

For iteration, `configs/dev.yaml` uses the same boundaries and keeps a uniform
10 % row sample of each window (seed 42). Sampling thins entity history, so
dev-split numbers are for smoke tests and plumbing, never for decisions.

## Consequences

- `configs/split.yaml` is the single source of the boundaries; `fraud.data.split`
  reads it; `tests/test_split.py` asserts ordering, disjointness and sizes.
- The test window is evaluated once per model candidate, at the end (project
  rules §8).
- Any feature that reads `TransactionDT` must be argued in via the leakage
  audit; the split module is the only place the raw timestamp is consumed as
  time.
