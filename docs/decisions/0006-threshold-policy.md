# ADR 0006 — Cost model and operating threshold

**Date:** 2026-09-14 · **Status:** accepted

## Context

The model outputs a score; the business needs a decision. The decision has
asymmetric costs, and neither Kaggle nor the data provider states them, so
they are assumed here, written down, and varied in a sensitivity table. The
choice is about a *policy*, and the policy must be defensible in the model
card, not optimal for an unknown business.

The consumer is a card-not-present merchant-side system (ADR 0001): a
declined transaction loses the sale; an approved fraud loses the goods and
the chargeback.

## Options

1. **Fixed threshold 0.5.** Arbitrary; ignores costs and the 3.5 % prior.
2. **Maximise F1.** Treats false positives and false negatives as equally
   costly per count. Wrong for fraud, where a missed $900 fraud and a
   declined $9 coffee are not the same.
3. **Amount-aware expected-cost minimisation.** Assign a cost to each error
   as a function of the transaction amount and pick the threshold that
   minimises total cost on the decision set.
4. **Target a precision or recall.** Simple to explain, but the target has
   to come from somewhere — usually from a cost argument anyway.

## Decision

**Option 3, with a single global threshold on the probability** (not a
per-transaction expected-value rule — see consequences).

Cost assumptions, per transaction of amount *a*:

| error | cost | rationale |
|---|---|---|
| false negative (approved fraud) | *a* + 15 | full loss of the amount (merchant liability for CNP chargebacks) plus a chargeback fee |
| false positive (declined legitimate) | 0.10 · *a* + 2 | lost margin on the sale plus the cost of the friction (support contact, some churn) |
| true positive / true negative | 0 | — |

Selection procedure:

1. Score the **validation window** with the current best model. Validation is
   the decision set (§8); the threshold is a one-parameter decision made on
   it.
2. Sweep thresholds; at each, total cost = Σ FN cost + Σ FP cost using each
   transaction's own amount. Choose the minimiser. Report the whole curve
   and the cost at 0.5, at the F1-optimum, and at ±50 % on each cost
   assumption (sensitivity table).
3. **Cross-check on the training window only**, using out-of-fold predictions
   from the expanding-window CV folds (days 63 – 122), so there is a
   threshold that never saw validation. If the two disagree materially the
   validation curve is flat around the optimum and the choice is reported as
   a range.
4. The chosen value goes into `configs/threshold.yaml` and is the `threshold`
   every later evaluation and the API use. The test window is evaluated at
   this threshold as the final temporal reporting window (ADR 0002's
   consultation log records each time it was scored).

## Consequences

- Because costs scale with amount, the optimal *policy* would be per
  transaction (decline when p · (a + 15) > (1 − p) · (0.1 a + 2)). That
  requires probabilities that are calibrated across the amount range and is
  harder to reason about in a model card. A global threshold is chosen for
  this build; the per-transaction rule is documented as the natural next step
  and the calibration work in ADR 0007 is what would make it safe.
- Precision, recall, F1 and the confusion matrix are reported at the chosen
  threshold from here on (G5).
- The threshold is a property of *this* model's score distribution; it is
  re-selected whenever the shipped model changes.
