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
- The test window is a **final temporal reporting window**, not a blind
  holdout (see the consultation log below). No model, feature, parameter,
  threshold or calibration choice was made on it; but it has been reported
  on at several development milestones.
- Any feature that reads `TransactionDT` must be argued in via the leakage
  audit; the split module is the only place the raw timestamp is consumed as
  time.

## Educational random split (added 2026-09-15, experiment E010)

G3 requires a written justification for any random or stratified split.
This one is a **control**: `scripts/split_comparison.py` draws a stratified
random split over the pooled train + validation rows (the test window is
excluded) and trains the same configuration, so the number a random split
*would have reported* can be put next to the temporal one. It is logged
under a separate MLflow experiment (`fraud-methodology`), never compared
against candidates, and never used to choose anything. Its purpose is to
show, on this data, why validation methodology matters.

## Test-window consultation log (amended 2026-09-15)

The project rules say the test window is evaluated once per model candidate
and is never used to choose between experiments. The second half was kept;
the first half, read literally, was not — and the documentation used to say
"scored once, at the end", which is not an accurate description once the
same later window has been reported on repeatedly during development. The
honest statement:

> **No model was selected based on test performance; however, multiple
> development milestones were reported on the same later window, so it is
> no longer a strictly blind holdout. Test numbers should be read as
> "one month further out than validation", with a small optimistic bias
> from having been seen, not as an unbiased estimate from an unseen set.**

Every consultation, in order (all on the same 85,430 rows, days 153–183):

| # | when | what was scored on test | decision made on it? |
|---|---|---|---|
| 1 | after E008 accepted | `evaluate_test` for `xgb_v2_tuned` (E008 + calibration + 0.08) | no |
| 2 | after E016 accepted | `evaluate_test` for `xgb_f5_interactions` | no |
| 3 | review-policy work | `review_policy --with-test` for E016 (policy table on test) | no — but it *informed* the recommendation "review by rank, not threshold" |
| 4 | final bundle | `final_report` + error analysis for E016 (same predictions as #2) | no |
| 5 | after E022 accepted | `evaluate_test` for `xgb_f5_capacity` | no — and E022 was shipped *despite* a weaker test transfer, precisely to avoid selecting on test |
| 6 | E022 re-ship | `review_policy --with-test`, `final_report` for E022 (same predictions as #5) | no |
| 7 | rank-policy work | one test day (day 170) scored through the API to show review-volume drift | no |
| 8 | monitoring demo | one test day (day 175) scored through the API and reported with labels (`reports/monitoring/demo_test_day175.md`) | no |

Three model candidates and two policy checks, so the window has been seen
in five distinct forms. The one place test evidence changed *documentation*
rather than a model: the "review by rank" recommendation (#3) was motivated
partly by the drift visible on test. That is a methodological finding
about policy shape, not a model choice, but it is recorded here as a test
consultation.

Going forward: any further candidate is reported on this window with the
caveat above, or — cleaner — a validation protocol with a later horizon is
adopted first (PROGRESS.md follow-up #1) so that the reporting window can
be re-frozen for a genuinely single final look.
