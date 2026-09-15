# ADR 0003 — Primary metric and imbalance handling

**Date:** 2026-09-14 · **Status:** accepted

## Context

- 3.5 % positives. Accuracy of a constant "not fraud" classifier is 96.5 %.
- The consumer is a threshold decision (approve / review / decline) whose
  operating point will sit in the high-score tail. Performance in the tail is
  what matters; performance on the easy 90 % is not.
- ROC-AUC is dominated by the ranking of the many negatives and moves little
  when tail precision changes. Precision–recall summarises the tail directly.

## Options

- ROC-AUC (Kaggle's competition metric).
- PR-AUC / average precision.
- Recall at a fixed precision, or precision at a fixed recall.
- F1 at a chosen threshold.

## Decision

**Primary metric: average precision (PR-AUC) on the validation window.**
Every experiment is compared on this number.

Always reported alongside it (project rules G5): ROC-AUC; precision, recall,
F1 and the confusion matrix at the current operating threshold (0.5 until
ADR 0006 sets one from costs); recall at precision ≥ 0.90 and precision at
recall ≥ 0.50 as tail summaries; later, Brier score and a reliability diagram
when calibration is assessed (ADR 0007).

A change is accepted when validation PR-AUC improves by **≥ 0.01** at the
default seed, or — for any smaller delta — when it is re-run on two more
seeds and the **paired mean improvement is ≥ 0.005 with every pair
positive**. Basis: E004 measured a seed standard deviation of 0.002 (E012
confirmed it for the tuned model), so 0.005 paired is ~3 sd.

*Amendment 2026-09-15 (E016):* the original wording only allowed the seed
re-run for deltas in 0.005 – 0.01; a +0.0035 result would have been called
noise without looking. The amended rule lets any change be seed-paired and
is stricter on the outcome (all pairs positive). Re-applied to every earlier
decision (E005 +0.0024, E007 −0.0019, E012 +0.0017, E017 −0.0005) it
changes none of them.

**Class imbalance: nothing by default.** Average precision and ROC-AUC are
ranking metrics; re-weighting or resampling changes the score scale and the
threshold, not the ranking a tree model learns from the same splits, and it
distorts probabilities that calibration would later have to undo. Weighting
(`class_weight`, `scale_pos_weight`) is allowed as a logged experiment with
a hypothesis; resampling (SMOTE, under-sampling) is out until a measured
reason exists.

## Consequences

- `fraud.evaluate.metrics` computes the full set from `(y_true, y_score,
  threshold)`; no evaluation prints a single number.
- The threshold used for precision/recall/F1 is a config value, defaulting to
  0.5 and replaced by ADR 0006's operating point.
