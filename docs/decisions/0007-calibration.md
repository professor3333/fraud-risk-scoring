# ADR 0007 — Calibration

**Date:** 2026-09-14 · **Status:** accepted

## Context

The system's stated output is a *probability* of fraud (ADR 0001), and the
cost model in ADR 0006 is expressed in expected-cost terms whose natural
extension is a per-transaction rule `p · cost_FN > (1 − p) · cost_FP`. Both
need `p` to mean what it says. Boosted trees trained with log-loss on a 3.5 %
positive rate and then evaluated on a later window are typically
over-confident in the tail and drift with the base rate.

Ranking metrics (PR-AUC, ROC-AUC) do not care about calibration; the
threshold, the API's probability, and the model card do.

## Options

1. **No calibration.** Report the raw score; call it a score, not a
   probability. Simplest; contradicts the intended use.
2. **Platt scaling** (logistic on the score). Two parameters; assumes a
   sigmoid shape; robust with little data.
3. **Isotonic regression.** Non-parametric monotone map; fits arbitrary
   miscalibration; needs a few thousand held-out rows to be stable; ties
   scores within a step, so ranking metrics can move slightly.
4. **`CalibratedClassifierCV`.** Convenient, but it either refits the model
   per fold (an ensemble the API would not be serving) or uses folds we do
   not control for time order.

Where to fit the calibrator matters more than which one:

- On validation: standard practice elsewhere, but G2 says nothing is fit on
  validation, and it would leave no untouched window on which to *assess*
  calibration before the single test evaluation.
- On training rows the model was fit on: over-confident scores, useless map.
- **On out-of-fold training-window scores** from the expanding-window folds
  (days 63 – 122, ≈ 200k rows, ≈ 7k positives): held out from the fold
  models, strictly inside the training window, plentiful enough for
  isotonic. The map is then applied to the final model, which was refit on
  all training days — a small mismatch (the final model is slightly stronger
  than the fold models) that is the price of respecting G2.

## Decision

**Isotonic regression fit on out-of-fold training-window scores, wrapped with
the fitted pipeline as one served object (`fraud.pipeline.calibrated.
CalibratedModel`).** Applied only if it improves Brier score and expected
calibration error on the validation window without moving PR-AUC by more
than 0.01; otherwise option 1 (raw score, labelled as such) and the reason
recorded here.

Assessment on validation reports, for raw and calibrated: Brier score (and
the prior's Brier as the floor), ECE over 15 quantile bins, and a reliability
diagram on log axes (the interesting region is 0.5 – 50 %).

## Consequences

- `scripts/calibrate.py` produces the OOF scores, fits the map, assesses,
  and saves `models/<run>_calibrated.joblib`. The threshold in ADR 0006 is
  selected on *calibrated* probabilities so that it is a probability the
  model card can state.
- The API serves the calibrated object; `/predict` returns the calibrated
  probability. Rankings are preserved (monotone map) up to isotonic ties.
- Calibration is a property of the training window's base rate. If the live
  fraud rate moves, the probabilities are off by roughly that ratio — a
  monitoring item for the model card.
