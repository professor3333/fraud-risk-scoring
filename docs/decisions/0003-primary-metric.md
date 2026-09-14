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

A change is accepted when validation PR-AUC improves and the improvement is
larger than the run-to-run noise measured by the reproducibility test and
the seed-variation experiment; this rule is refined once that noise is
measured.

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
