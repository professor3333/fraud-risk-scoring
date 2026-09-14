# Experiment log

One entry per experiment. Hypothesis and expected direction are written
**before** the run; results and interpretation after. All numbers are on the
**validation** window (ADR 0002) unless marked otherwise. Primary metric is
average precision (ADR 0003). MLflow experiment `fraud-baselines` holds the runs.

## E001 — Constant-rate baseline

- **Hypothesis:** a model that predicts the training fraud rate for every row
  has PR-AUC equal to the validation positive rate and ROC-AUC 0.5. This is the
  floor every later model must clear.
- **Config:** `configs/model/constant.yaml`, split `configs/split.yaml`.
- **Expected:** PR-AUC ≈ 0.034, ROC-AUC = 0.5, recall = 0 at threshold 0.5.
- **Result** (run `6aa9a1d2`): validation PR-AUC **0.0339**, ROC-AUC 0.500,
  precision / recall / F1 = 0 / 0 / 0 at 0.5, confusion tp 0 · fp 0 · fn 2,884
  · tn 82,160. Train PR-AUC 0.0352.
- **Interpretation:** as expected — the floor is the positive rate. Any model
  whose validation PR-AUC is not clearly above 0.034 has learned nothing.

## E002 — Logistic regression on raw provider columns

- **Hypothesis:** a linear model on median-imputed, scaled numeric columns plus
  one-hot low-cardinality categoricals captures a useful share of the signal
  (product, card type, identity presence, `V` blocks) but is limited by the
  non-linear, interaction-heavy structure of fraud.
- **Config:** `configs/model/logreg.yaml` (C = 1.0, lbfgs, 2000 iterations).
- **Expected:** validation PR-AUC 0.30 – 0.45, ROC-AUC 0.80 – 0.87, a visible
  train/validation gap (dev split showed 0.53 vs 0.39). Precision at
  threshold 0.5 high, recall low — the model is confident only on the easy
  positives.
- **Result** (run `98a6a1eb`): validation PR-AUC **0.4020**, ROC-AUC 0.8419;
  at threshold 0.5 precision 0.719, recall 0.237, F1 0.357; confusion tp 684 ·
  fp 268 · fn 2,200 · tn 81,892; recall at precision ≥ 0.90 = 0.118. Train
  PR-AUC 0.4925 → gap 0.090. Wall time ≈ 16 min on the full training window
  (dense 470-column matrix; memory-bound).
- **Interpretation:** inside the expected band and ~12× the constant floor,
  so the raw provider columns carry substantial linear signal. The gap says
  the model overfits mildly even with no engineered features — 339 `V`
  columns and their indicators are a lot of freedom for a linear model. At
  0.5 it flags only 952 of 85k transactions and is right on 72 % of them:
  high precision, low recall, the usual shape for an uncalibrated linear
  baseline on rare positives. This is the number XGBoost has to beat
  (Stage 3), and the training-time cost argues for a sparse or reduced
  feature matrix if the baseline is ever refit routinely.
