# Experiment log (long form)

The four-column summary is `docs/EXPERIMENT_LOG.md`.

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


## E009 — Ablation and importance of the current best

- **Hypothesis:** the `V` block, which the EDA shows is engineered and
  block-structured, will dominate split gain but be partly redundant with
  the raw columns; `C*` and `card*` will be the groups the model cannot do
  without; `has_identity` will contribute nothing (product proxy).
- **Config:** `scripts/ablation.py` on E008 — gain, group permutation on
  validation, and one refit per removed group (MLflow `fraud-ablation`).
- **Expected:** removing `V` costs 0.01 – 0.03; removing `C` or `card`
  costs ≥ 0.03; `has_identity` ≈ 0.
- **Result:** `docs/ablation.md`. `V`: gain 76 %, permutation −0.19,
  **ablation −0.005**. `card` −0.053, `C` −0.049, frequency −0.014, `M`
  −0.014, `addr` −0.011. `has_identity` 0.000 on every method; removing it
  or `dist` gives +0.005 (noise band).
- **Interpretation:** the hypothesis on `C`/`card`/`has_identity` held; `V`
  is *more* redundant than expected — the data does almost as well without
  it. Gain and permutation describe the fitted model; ablation describes the
  data. A 131-column model without `V` is a candidate follow-up experiment.

## Final — test-window report of E008 + calibration + threshold 0.08 (consultation #1)

- **Procedure:** `scripts/evaluate_test.py`, run once (MLflow `fraud-final`).
  No decision was made on the test window.
- **Result:** test PR-AUC **0.553** (validation 0.616), ROC-AUC 0.910
  (0.929); at 0.08 precision 0.275 / recall 0.697 / F1 0.394; recall at
  P ≥ 0.90 0.237; top-500-per-day recall 0.822; Brier 0.0219, ECE 0.0060;
  cost 275k vs 477k approve-all vs 363k at 0.5.
- **Interpretation:** a 0.06 drop one month further out, with recall at the
  threshold holding and precision falling — score-scale drift rather than
  lost ranking ability. Numbers in `README.md` and `docs/model_card.md`.

## E010 — Random stratified split vs the frozen temporal split (educational)

- **Hypothesis:** a random stratified split over the same rows will report a
  *higher* validation PR-AUC than the temporal split for the same model,
  for two reasons that are both invisible to it: rows from the same account
  land on both sides (and the label propagates within accounts, ADR 0001),
  and the validation rows come from the same months as training, so the
  drift that the temporal split measures (0.06 PR-AUC from validation to
  test) is hidden. The random number is a control, not a candidate (G3;
  ADR 0002 §"Educational random split").
- **Config:** `scripts/split_comparison.py` with E008's config; pool =
  temporal train ∪ validation (505,110 rows), stratified random split of
  the same sizes, seed 42. The test window is not in the pool.
- **Expected:** random validation PR-AUC 0.10 – 0.25 above the temporal
  0.6155; a smaller train/validation gap.
- **Result** (`reports/split_comparison/xgb_v2_tuned.json`, MLflow
  `fraud-methodology`): same model, same rows, same sizes —

  | split | val PR-AUC | val ROC-AUC | recall @ P ≥ 0.90 | train PR-AUC | gap |
  |---|---:|---:|---:|---:|---:|
  | temporal (frozen) | 0.6155 | 0.929 | 0.330 | 0.918 | 0.303 |
  | random stratified | **0.8102** | 0.963 | **0.654** | 0.912 | 0.102 |

- **Interpretation:** the random split over-reports PR-AUC by **0.19** and
  recall at 90 % precision by 2×, and makes the model look far less
  overfit (gap 0.10 vs 0.30). Nothing about the model changed; only what
  the validation rows share with the training rows did: the same accounts
  (whose label propagates, so memorising an account *is* predicting its
  label) and the same weeks (so drift never shows). A project selecting on
  the random number would ship a model expecting 0.81 and get 0.55 on the
  next month (the test result). Every decision in this repository was made
  on the temporal number; this run is a control and is not in the results
  table's comparison set.


## E002b — Logistic regression on a small interpretable subset

- **Hypothesis:** a linear model on 20 numeric columns (amount, billing
  region, the 14 `C` counts, `D1`/`D10`/`D15`) plus one-hot product, card
  type and both e-mail domains recovers a substantial share of E002's
  signal — the `C` counts and product/e-mail presence are where the
  marginal signal sits in the EDA — at a fraction of the cost. Precision /
  recall / F1 at 0.5 are recorded for completeness only; the threshold is
  chosen in its own component (ADR 0006).
- **Config:** `configs/model/logreg_small.yaml`; features
  `baseline_small` (26 raw columns → ~150 encoded).
- **Expected:** validation PR-AUC 0.20 – 0.30 (half of E002's 0.402),
  ROC-AUC 0.75 – 0.82; seconds rather than minutes.
- **Result** (run `309924ff`): validation PR-AUC **0.2884**, ROC-AUC 0.795;
  at 0.5 precision 0.895 / recall 0.083 / F1 0.151 (provisional threshold),
  confusion tp 238 · fp 28 · fn 2,646 · tn 82,132; recall at P ≥ 0.90 =
  0.082. Train PR-AUC 0.253 — *below* validation. Wall time 22 s.
- **Interpretation:** 26 raw columns give 72 % of the 423-column linear
  model's PR-AUC (0.288 vs 0.402) in 1/40 of the time, and with no
  overfitting at all (the train score is lower than validation, which says
  the validation month is slightly easier for these columns). The signal a
  linear model can use lives mostly in the `C` counts and the product /
  e-mail / card-type levels. Nearly all of E002's extra 0.11 comes from the
  `V` block and its missing-indicators — consistent with the later ablation
  (`V` matters to models that can use it, but is replaceable). Neither
  linear model is a candidate; both are reference points for the trees.


## E011 — Median imputation + missing indicators in front of the trees

- **Hypothesis:** XGBoost's native missing-value routing already learns a
  direction for NaN at every split, so replacing NaN by the training median
  and adding indicator columns gives the trees the same information in a
  clumsier form. Expect no gain; a small loss is possible because the
  median hides *which* value was missing inside a `V` block that is 86 %
  null.
- **Config:** `configs/model/xgboost_v2_tuned_impute.yaml` = E008 with
  `preprocessing: linear`. One change.
- **Expected:** validation PR-AUC within ±0.005 of 0.6155. Not accepted
  either way unless ≥ +0.01.
- **Result:** validation PR-AUC **0.6129** (−0.0026), ROC-AUC 0.933, recall
  at P ≥ 0.90 0.326. Wall time 3 min vs 1.5 min (the indicator columns
  double the dense matrix).
- **Interpretation:** as hypothesised — no gain, inside noise, twice the
  cost. Trees do not need imputation here; NaN routing stays the default
  for the tree path. The option remains available (`preprocessing: linear`
  in a model config) so the claim can be re-checked on another dataset.


## E012 — Rare one-hot levels grouped into one "infrequent" column

- **Hypothesis:** the 22 one-hot columns have ≤ 5 levels each, and only
  `card6` carries genuinely rare levels (`charge card` 15 rows, `debit or
  credit` 30 rows). Grouping levels with < 100 training rows — and routing
  unseen levels into the same column instead of all-zeros — is a safer
  serving contract but should change the score by nothing measurable.
- **Config:** `configs/model/xgboost_v2_tuned_rare.yaml`, feature set
  `v2_freq_rare` (`rare_min_frequency: 100`). One change.
- **Expected:** validation PR-AUC within ±0.003 of 0.6155.
- **Result:** validation PR-AUC 0.6214 (+0.0059) at seed 42 — in the re-run
  band. Seed pairs (E008 vs E012): seed 1 0.6160 vs 0.6144 (−0.0016), seed 2
  0.6119 vs 0.6126 (+0.0007). Mean paired delta **+0.0017**.
- **Interpretation:** noise, as expected. Rare grouping is score-neutral
  here because the one-hot columns have almost no rare levels. It *is* the
  better serving contract (unseen levels share a learned column instead of
  vanishing), so it is the recommended setting at the next retrain; the
  shipped model is not rebuilt for a zero-gain change. The E008 seed pairs
  measured here (0.6155 / 0.6160 / 0.6119, sd 0.002) confirm the noise floor
  from E004 for the tuned model.


## Feature sets F1 – F7 (E013 – E017)

Each set is one change on top of the current best (E008, validation PR-AUC
0.6155; seed pairs 0.6160 / 0.6119). Same acceptance rule as always. All
derived columns are row-local and computed inside the pipeline
(`fraud.features.rowwise`), so serving needs nothing new.

### E013 — F1: missing-field counts

- **Hypothesis:** EDA §4b shows address and match fields are missing far
  more often for fraud, and identity fields less often. The trees can
  already split on each column's NaN, so a *count* of missing fields adds
  only a coarse summary; expected small or no gain.
- **Config:** `xgboost_f1_missingness.yaml` (+ `n_missing_transaction`,
  `n_missing_identity`, `n_missing_total`).
- **Expected:** −0.003 … +0.008.
- **Result:** validation PR-AUC 0.6149 (−0.0006). **Neutral.** The trees
  already see every column's NaN; the counts add nothing.

### E014 — F2: amount structure

- **Hypothesis:** the log is a monotone transform (invisible to trees); the
  integer / cents decomposition and "round amount" flag are new information
  the raw amount does not expose as a split. EDA found the cents pattern
  not discriminative *marginally* (47 % vs 48 % with cents); any gain must
  come from interactions with product or card type. Expected ≈ 0.
- **Config:** `xgboost_f2_amount.yaml` (+ `amt_log`, `amt_integer_part`,
  `amt_fraction`, `amt_is_round`, `amt_cents_digits`).
- **Expected:** −0.003 … +0.006.
- **Result:** validation PR-AUC 0.6150 (−0.0005). **Neutral**, as the EDA
  suggested: cents patterns carry no signal here, and the log is invisible
  to trees.

### E015 — F4: e-mail flags and provider families

- **Hypothesis:** `same_email_domain` is a genuinely new bit (purchaser =
  recipient); presence flags duplicate the frequency table's `<missing>`
  share; provider families are a coarsening of the 59/60-level frequency
  encoding. Expected small gain from `same_email_domain` at most.
- **Config:** `xgboost_f4_email.yaml`.
- **Expected:** −0.002 … +0.008.
- **Result:** validation PR-AUC 0.6097 (−0.0058). **Rejected** — the only
  set with a clearly negative sign. The provider-family coarsening gives
  the trees a lower-resolution copy of information the 59/60-level
  frequency table already carries, and the extra split candidates cost a
  little. `same_email_domain` alone was not isolated; it could be revisited
  as a single column if e-mail is ever revisited.

### E016 — F5: card × address / e-mail / card-type keys, frequency-encoded

- **Hypothesis:** `card1` alone has a median of 2 start-days per value
  (EDA §6), i.e. it is not one account; `card1+addr1` and
  `card1+addr1+P_emaildomain` are closer to an account, and their
  training-window frequency says how established that account is. This is
  the F5 + F6 combination; the host confirms multiple rows per account.
  Expected the largest gain of the row-local sets.
- **Config:** `xgboost_f5_interactions.yaml` (four keys under `frequency`).
- **Expected:** +0.005 … +0.02.
- **Result** (run `xgb_f5_interactions`): validation PR-AUC **0.6190**
  (+0.0035), ROC-AUC 0.932, recall at P ≥ 0.90 0.329. Seed pairs vs E008:
  seed 1 0.6221 vs 0.6160 (+0.0061), seed 2 0.6205 vs 0.6119 (+0.0086).
  Mean paired delta **+0.0061**, positive on every seed.
- **Interpretation:** **accepted** under ADR 0003 as amended today (paired
  mean ≥ 0.005 and all pairs positive; the seed-42 delta alone would not
  have triggered a re-run under the old wording). The composite keys give
  a notion of "how established is this card *at this address / with this
  e-mail*" that neither `card1` frequency nor the provider's counts carry.
  Smaller than the Kaggle folklore promises, because those gains came
  from counting over the whole dataset. New current best: E016,
  feature set `f5_interactions`.

### E017 — F7: history on `card1+addr1` with std / max (only after F0–F6)

- **Hypothesis:** E007 used `(card1, addr1, day − D1)` with count, recency,
  mean and ratio and found nothing. A coarser entity (`card1+addr1`, no
  `D1`) with the same strictly-earlier computation plus expanding std and
  max asks "is this amount unusual for this card-at-this-address". Expected
  ≈ 0 again, because the provider's `C*`/`D*` columns already summarise
  the card's past; run to close the question for this entity definition.
- **Config:** `xgboost_f7_history.yaml`.
- **Expected:** −0.005 … +0.01.
- **Result:** validation PR-AUC 0.6185 (−0.0005 vs E016). **Rejected.** Same
  verdict as E007 under a different entity definition and with std / max
  added: once the history is restricted to what a live system can compute,
  the provider's `C*` / `D*` columns already contain it. Two entity
  definitions, eight features, no gain — the question is closed for this
  dataset. The module stays as the reference implementation.

### Feature-set summary

| set | experiment | Δ validation PR-AUC | verdict |
|---|---|---:|---|
| F0 raw | E003 | — | baseline |
| F1 missing counts | E013 | −0.001 | neutral |
| F2 amount structure | E014 | −0.001 | neutral |
| F3 time of day | E005 | +0.002 paired | rejected |
| F4 e-mail flags / families | E015 | −0.006 | rejected |
| F5 card × address / e-mail keys, frequency-encoded | E016 | **+0.006 paired** | **accepted** |
| F6 frequency encoding | E006 | +0.009 paired | accepted |
| F7 entity history (two definitions) | E007, E017 | −0.002, −0.001 | rejected |


## E018 — Cumulative feature ladder and family cuts

- **Hypothesis (ladder):** accumulating every set in order F0 → F7 with the
  tuned parameters should track the per-set results: flat through F1 – F4,
  a step up at F5/F6, flat at F7. If "everything" lands *below* F0+F5+F6,
  accumulation is costing something (extra split candidates at depth 8).
- **Hypothesis (cuts):** most of the score is Vesta's engineering. Removing
  `V` from the shipped set costs ≈ 0.005 (as in E009's ablation); removing
  all of `C`, `D`, `M`, `V` costs far more; a transaction-only model without
  identity and without `V` loses a further ≈ 0.01. My own contribution
  (frequency tables and composite keys) is the E006 + E016 total ≈ 0.015.
- **Config:** `scripts/feature_ladder.py --model configs/model/xgboost_f5_interactions.yaml`
  (E008 parameters throughout; MLflow `fraud-feature-sets`).
- **Result:** `docs/feature_sets.md`. Ladder: F1 – F4 within ±0.004, F5
  +0.013, F6 +0.009, F7 −0.006; cumulative "+F6" 0.6255. Cuts: raw
  transaction-only 0.586 → raw +identity 0.591 → F0 0.601; shipped
  transaction-only 0.604, shipped-minus-`V` 0.621, shipped 0.619,
  shipped-minus-`C/D/M/V` **0.433**.
- **Interpretation:** the ladder confirms the one-at-a-time verdicts; the
  cuts put numbers on the question: Vesta's `C`/`D`/`M`/`V` are worth 0.19,
  my frequency / key features 0.02 – 0.03, and `V` specifically is worth
  0.01 on raw columns and nothing on the shipped set. Two apparent
  improvements were seed-paired next (E019, E020).

## E019 — Cumulative F0 – F6 as a candidate

- **Hypothesis:** the ladder's "+F6" (0.6255) beats E016 (0.6190) by more
  than the band; if real, accumulating F1 – F4 is worth keeping.
- **Config:** `configs/model/xgboost_f6_cumulative.yaml`, seeds 42 / 1 / 2.
- **Result:** 0.6255 / 0.6156 / 0.6214 vs E016 0.6190 / 0.6221 / 0.6205 →
  paired **+0.0003**.
- **Interpretation:** rejected; a lucky seed. Consistent with E013 – E015.

## E020 — Shipped set without `V` as a candidate (compact model)

- **Hypothesis:** with frequency tables and composite keys present, the
  339 `V` columns are replaceable; a 100-input model matches the shipped
  score.
- **Config:** `configs/model/xgboost_f5_noV.yaml`, seeds 42 / 1 / 2.
- **Result:** 0.6214 / 0.6159 / 0.6168 vs E016 → paired **−0.0025**; ROC-AUC
  0.931; fit time roughly a quarter.
- **Interpretation:** score-neutral (within noise, slightly negative), so
  not shipped under the acceptance rule; recorded as the compact option
  for when serving cost matters. This is the strongest statement the
  project can make about `V`: optional once entity frequency is modelled.


## E022 — More capacity, as the sweep points (depth 12, mcw 1, 1600 trees)

- **Hypothesis:** E021 shows validation PR-AUC rising monotonically with
  depth (to 12), with smaller `min_child_weight` (to 1) and with trees (to
  ~1,700) despite a widening train/validation gap. Combining those three
  moves should beat E016 by more than any single one (+0.0098 for depth 12
  alone).
- **Config:** `configs/model/xgboost_f5_capacity.yaml`; seeds 42 / 1 / 2 vs
  E016 (0.6190 / 0.6221 / 0.6205).
- **Expected:** +0.01 – 0.02 paired; fit time ≈ 3× the anchor.
- **Result:** validation PR-AUC **0.6367 / 0.6359 / 0.6399** vs E016
  0.6190 / 0.6221 / 0.6205 → paired **+0.017**, every seed positive. ROC-AUC
  0.9325, recall at P ≥ 0.90 0.334, Brier 0.0185 / ECE 0.0037 after sigmoid
  calibration (raw 0.0202 / 0.0167). Threshold re-check: validation 0.085,
  OOF 0.08, flat band 0.065 – 0.135 → **0.08 stays**. Policy re-check: block
  0.42 unchanged, 200/day review threshold 0.062, recall 0.776, cost 157k.
  Artifact 41 MB; fit ≈ 4 min.
- **Interpretation:** **accepted and shipped (v0.3.0).** The advantage over
  E016 holds across the whole validation month (+0.020 / +0.014 / +0.018 by
  10-day block), so there was no validation-side reason to doubt it. Its
  test-window report (below) is a caveat, recorded, not a selection.

## Final — test-window report of E022 + calibration + threshold 0.08 (consultation #5)

- **Procedure:** `scripts/evaluate_test.py`, once (MLflow `fraud-final`).
- **Result:** test PR-AUC **0.5610** (E016: 0.5566), ROC-AUC 0.907 (0.910);
  at 0.08 precision 0.262 / recall 0.691 / F1 0.380 (E016: 0.275 / 0.717 /
  0.397); recall at P ≥ 0.90 0.243 (0.256); top-500-per-day recall 0.801
  (0.816); Brier 0.0214, ECE 0.0059; cost 275k vs 481k approve-all (E016:
  262k). Policy at 200/day: recall 0.727, block precision 0.727, 273
  reviewed/day.
- **Interpretation:** the +0.017 validation gain became +0.004 on the
  ranking metric two months out, and the operating-point metrics moved
  slightly the other way. The higher-capacity model memorises the training
  window harder and transfers a little worse — the caveat written into the
  progression document before this number existed. The decision rule was
  applied as written (validation decides; test reports). What this exposes
  is a *methodology* limit: a validation window adjacent to training
  measures one-month transfer, and the second month is where capacity's
  cost appears. Recorded as the first item in `PROGRESS.md`'s follow-ups: a
  drift-aware validation protocol (a gap, or a later window) decided in an
  ADR *before* any further model selection — not a re-decision of E022 on
  the test number.


## E023 — Rolling temporal backtests: does the horizon change the winner? (ADR 0008)

- **Hypothesis:** on the adjacent horizon (gap 0) the ranking is E022 >
  depth-10 > E016, as on ADR 0002's window. At gaps ≥ 30 days E022's
  advantage shrinks or reverses, because the higher-capacity model
  memorises the training window harder and transfers worse — the pattern
  the reporting window showed (+0.017 → +0.004). The decay
  (adjacent − robust) is expected to be largest for E022.
- **Config:** `scripts/backtest.py` with `configs/backtest.yaml` on
  `xgboost_f5_interactions` (E016), `xgboost_f5_depth10`, `xgboost_f5_capacity`
  (E022); MLflow `fraud-backtest`. Development days ≤ 152 only.
- **Expected:** robust (gap ≥ 30) means within ~0.01 of each other, E022's
  lead ≤ +0.005; if E016 or depth-10 wins the robust mean by the ADR 0003
  rule (seed-paired), that candidate becomes the shipped model.
- **Result** (`docs/backtest.md`, `reports/backtest/`): E022 wins all ten
  windows at seed 42; its lead over E016 is +0.013 adjacent, +0.007 at gap
  30, +0.006 at gap 60. Robust mean paired over seeds 42 / 1 / 2:
  +0.0065 / +0.0042 / +0.0102 → **+0.0070**, all positive. Depth 10 sits
  between the two everywhere.
- **Interpretation:** the hypothesis was half right. The advantage of the
  higher-capacity model *shrinks* with horizon (about halves), which is the
  shape the reporting window showed; it does *not* reverse. **E022 stands
  as the shipped model on validation-only evidence** under the protocol
  that would have flagged the shrinkage before shipping. The decay column
  is now part of every candidate comparison.

## E024 — Monthly retraining lifecycle, simulated offline (`docs/retraining.md`)

- **Hypothesis:** a model one month staler than the freshest possible
  loses materially on the next month; a champion/challenger cycle with a
  +0.005 promotion margin will promote the fresh challenger every month on
  this data.
- **Config:** `scripts/retrain.py` with `configs/retrain.yaml` — cut-offs at
  days 120 and 150 (development data only), challenger recipe E022,
  calibration on OOF folds inside the challenger's training data, block
  threshold re-selected on the validation month at the 80 % precision bar,
  artifact frozen with a golden.
- **Expected:** the challenger beats the incumbent by ≥ 0.05 at cut-off 150.
- **Result:** cut-off 120 (bootstrap): challenger trained through day 90
  scores 0.6146 on days 91–120; block threshold 0.470. Cut-off 150: the
  incumbent scores **0.5216** on days 121–150, the challenger trained
  through day 120 scores **0.6467** → +0.125, promoted; block threshold
  re-selected to 0.425.
- **Interpretation:** one month of staleness costs 0.125 PR-AUC on the
  next month — the same magnitude the backtest's gap-30 windows show
  (~0.53 vs ~0.64). Monthly retraining is not a nicety on this data; it is
  worth more than every feature experiment combined. The block threshold
  moved 0.47 → 0.425 between cycles, which is why it is re-selected rather
  than frozen.

