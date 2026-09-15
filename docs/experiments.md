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

## E003 — XGBoost on the same raw feature set

- **Hypothesis:** gradient-boosted trees capture the non-linear and
  interaction structure (product × amount × `C`/`V` patterns, informative
  missingness) that the linear model cannot, on exactly the same columns.
  The only change vs E002 is the model and the preprocessing it needs (no
  imputation or scaling; NaN routed natively).
- **Config:** `configs/model/xgboost.yaml` — 600 trees, η 0.05, depth 6,
  min_child_weight 5, subsample 0.8, colsample 0.6, hist. Fixed tree count,
  no early stopping, so validation is not consulted during fitting.
- **Expected:** validation PR-AUC 0.55 – 0.70 (a large step over 0.402),
  ROC-AUC ≥ 0.90, and a *larger* train/validation gap than the linear model —
  trees fit the training window harder. Recall at precision ≥ 0.90 should
  roughly double.
- **Result** (run `e7527953`): validation PR-AUC **0.5701**, ROC-AUC 0.9115;
  at 0.5 precision 0.848, recall 0.343, F1 0.489; confusion tp 990 · fp 177 ·
  fn 1,894 · tn 81,983; recall at precision ≥ 0.90 = 0.292. Train PR-AUC
  0.792 → gap 0.222. Wall time 43 s (vs 16 min for E002).
- **Interpretation:** +0.168 PR-AUC over E002 on identical columns — the
  non-linear structure is real and large. Recall at 90 % precision goes
  from 0.118 to 0.292: at the same false-positive budget the trees catch 2.5×
  the fraud. The gap is wide (0.22), as expected for 600 depth-6 trees on
  420k rows; it is the reason tuning and regularisation are an experiment
  later, not a reason to reject the model now. Accepted as the current best.

## E004 — Seed variation of E003 (noise floor)

- **Hypothesis:** the run-to-run spread from `random_state` alone (subsample,
  colsample) is small relative to the E002 → E003 step, and gives the minimum
  improvement a later change must exceed.
- **Config:** E003 with seeds 42, 1, 2, 3.
- **Expected:** validation PR-AUC standard deviation ≈ 0.002 – 0.005.
- **Result:** PR-AUC 0.5701 / 0.5718 / 0.5691 / 0.5737 — mean 0.5712,
  sd 0.0020, range 0.0046. Recall at P ≥ 0.90: mean 0.284, sd 0.008.
- **Interpretation:** noise is ~0.002 on the primary metric. **Acceptance rule
  from here on: a change is accepted when it improves validation PR-AUC by
  ≥ 0.01 (≈ 5 sd, > 2× the observed range) at the default seed; changes in
  the 0.005 – 0.01 band are re-run on two more seeds before a decision;
  smaller deltas are noise.** Recall at P ≥ 0.90 is too noisy (sd 0.008) to
  drive decisions on its own.

## E005 — Hour of day and weekday

- **Hypothesis:** the daily fraud cycle (10.6 % at "hour 7" vs 2.5 % mid-day,
  `docs/eda.md` §2) is not recoverable from any existing column (`D9` is the
  only clock-like column and is 87 % null), so `hour` adds signal; `weekday`
  is weak (flat fraud rate) and mostly along for the ride.
- **Config:** `configs/model/xgboost_v1_time.yaml` = E003 + feature set
  `v1_time` (`hour`, `weekday` derived in-pipeline). One change: the feature
  set.
- **Expected:** validation PR-AUC +0.01 – 0.03 over E003 (0.570). Accepted if
  ≥ +0.01 (ADR 0003).
- **Result** (run `b4d92e3f`): validation PR-AUC **0.5759** (+0.0058), ROC-AUC
  0.912, recall at P ≥ 0.90 0.284. In the 0.005 – 0.01 band, so re-run on
  seeds 1 and 2: 0.5739 (E003 seed 1: 0.5718, +0.0021) and 0.5685 (E003 seed
  2: 0.5691, −0.0006). Mean paired delta **+0.0024**.
- **Interpretation:** **rejected** — below the 0.005 floor on the seed-paired
  test. The hour-of-day signal is genuine in the EDA but the trees already
  recover it: `D8` carries fractional days and `D9` is hour / 24 where
  present, and the `V` blocks were engineered by the provider with time in
  hand. Lesson recorded: a strong marginal signal in EDA is not the same as a
  conditional gain over 400 provider features. The `Derive` step stays (it
  is the mechanism for future row-local features); `baseline_raw` remains
  the feature set of the current best (E003).


## E006 — Frequency encoding of twelve high-cardinality columns (ADR 0005)

- **Hypothesis:** "how common is this card / address / device / e-mail
  domain in the training population" separates rare-entity fraud from
  routine traffic in a way raw identifier values cannot. Encoded as
  training-window shares fit inside the pipeline; the raw numeric columns
  stay, so the delta measures the added value of frequency alone.
- **Config:** `configs/model/xgboost_v2_freq.yaml` = E003 + feature set
  `v2_freq`. One change: the feature set.
- **Expected:** validation PR-AUC +0.02 – 0.05 over E003 (0.570). Accepted if
  ≥ +0.01.
- **Result** (run `cf66ea02`): validation PR-AUC **0.5778** (+0.0077), ROC-AUC
  0.9186 (+0.007), recall at P ≥ 0.90 0.294. Band re-run: seed 1 0.5797
  (+0.0079), seed 2 0.5814 (+0.0123). Mean paired delta **+0.0093**, positive
  on every seed.
- **Interpretation:** **accepted** — smaller than hoped but consistent, and
  ROC-AUC moves too, so it is not a tail artefact. Frequency adds a notion
  the raw identifiers lack. The modest size suggests the provider's `C*`
  counts already encode much of "how established is this card"; a per-column
  ablation in Stage 5 will say which of the twelve earn their place. Current
  best: E006, feature set `v2_freq`.


## E007 — Entity history features from strictly earlier rows (ADR 0004)

- **Hypothesis:** velocity and deviation-from-habit on the reconstructed
  card entity (`ent_prior_count`, `ent_seconds_since_prev`,
  `ent_prior_amt_mean`, `ent_amt_ratio`, `ent_prior_count_1d`) carry signal
  the provider's static columns do not. The marginal fraud rate by prior
  count is only mildly increasing (2.0 % at 0 → 3.8 % at 11+), so the gain,
  if any, comes from interactions with amount and product, not from
  velocity alone. No label is used.
- **Config:** `configs/model/xgboost_v3_history.yaml` = E006 + feature set
  `v3_history`. One change: the five history columns.
- **Expected:** validation PR-AUC +0.01 – 0.04 over E006 (0.578). Accepted if
  ≥ +0.01. A jump far beyond that band would be a red flag to re-audit the
  computation for future information.
- **Result** (run `be7472fb`): validation PR-AUC **0.5816** (+0.0038), ROC-AUC
  0.920, recall at P ≥ 0.90 0.280. Below the floor, so rejected by rule;
  seeds 1 / 2 run for interpretation: 0.5735 (−0.0062), 0.5781 (−0.0033).
  Mean paired delta **−0.0019**.
- **Interpretation:** **rejected.** Restricted to what a real system can
  compute — label-free aggregates over strictly earlier rows of the entity —
  the reconstructed card id adds nothing over the provider's own `C*` / `D*`
  columns, which already summarise the card's past (that is presumably how
  the provider built them). The Kaggle gains attributed to "uid" features
  came from aggregates over the *whole* dataset (future rows, and via
  propagation, the label) evaluated on a test set sharing those entities;
  none of that is available at authorization. This is the single most
  important negative result in the project. The module and its G3 tests stay
  in the repo as the reference implementation should a legitimate variant
  (e.g. longer windows, distinct-merchant counts) be proposed later. Current
  best remains E006.


## E008 — Hyperparameter search with expanding-window CV

- **Hypothesis:** the E003/E006 parameters were a reasonable guess, not a
  tuned point. A 16-trial random search over depth, learning rate, tree
  count, child weight, subsampling and regularisation, scored by
  expanding-window CV inside the training window (folds: days 1–62 → 63–92
  and 1–92 → 93–122), finds a configuration that transfers to the validation
  window. The untuned parameters are scored on the same folds as a
  reference.
- **Config:** `configs/tuning/xgboost.yaml`; then the best trial refit on the
  full training window as `configs/model/xgboost_v2_tuned.yaml` and compared
  with E006 on validation. Feature set `v2_freq` throughout.
- **Expected:** best CV PR-AUC above the reference by 0.01 – 0.03; validation
  gain smaller than the CV gain (the search sees the folds, validation is
  untouched). Accepted if validation PR-AUC ≥ E006 + 0.01. The train/val gap
  is reported: a tuned model that only widens the gap is not an improvement.
- **Result:** search (MLflow `fraud-tuning`, 16 trials + reference,
  `reports/tuning/xgboost_trials.csv`): reference CV PR-AUC 0.607 ± 0.017;
  best trial 0.626 ± 0.017 (800 trees, η 0.05, depth 8, min_child_weight 5,
  subsample 0.8, colsample 0.5, λ 1, α 1). Every depth-4 trial scored below
  the reference; depth 8 filled the top three. Refit on the full training
  window (run `5ebc9a67`): validation PR-AUC **0.6155** (+0.0377 over E006),
  ROC-AUC 0.929, precision / recall / F1 at 0.5 = 0.867 / 0.374 / 0.522,
  recall at P ≥ 0.90 = 0.330. Train PR-AUC 0.918 → gap 0.30 (E006: 0.23).
  Wall time 86 s.
- **Interpretation:** **accepted** — nearly four times the bar, and the CV
  gain (+0.019) transferred to validation with room to spare, which says the
  folds are a faithful proxy for the later window. Deeper trees are what
  the data wanted: fraud here is interactions among many weak columns. The
  gap did widen; the learning curve (`reports/curves/`) is the check that
  it is benign — validation PR-AUC keeps rising with trees rather than
  turning over. Current best: E008 (`xgb_v2_tuned`).

