# Model card — fraud risk scoring, `xgb_f5_interactions` + sigmoid calibration

## Model details

- **Type:** gradient-boosted trees (XGBoost 3.4, `hist`), 800 trees, depth 8,
  η 0.05, min_child_weight 5, subsample 0.8, colsample 0.5, L2 1, L1 1;
  wrapped with a Platt (sigmoid-on-logit) calibrator. One sklearn object
  (`models/xgb_f5_interactions_calibrated.joblib`) does preprocessing,
  scoring and calibration; the API loads exactly that object.
- **Inputs:** the IEEE-CIS transaction record (393 provider columns) plus the
  identity record (40 columns) when one exists. Twelve high-cardinality
  columns and four row-local composite keys (`card1+addr1`,
  `card1+addr1+P_emaildomain`, `card1+card4`, `card1+card6`) are
  frequency-encoded from the training population;
  22 low-cardinality strings are one-hot encoded with `<missing>` as a
  level; missing numerics are left to the trees. `TransactionDT` and
  `TransactionID` are never inputs.
- **Output:** calibrated probability that the transaction is, or belongs to
  an account that becomes, reported as fraud; and a decision at 0.08.
- **Version:** `xgb_f5_interactions+sigmoid@<sha256 prefix of the artifact>`,
  returned by `/health` and `/predict`. MLflow: `xgb_f5_interactions` in
  `fraud-xgboost` (model), `fraud-calibration` (map), `fraud-final` (test
  evaluation).

## Intended use

Score a card-not-present transaction at authorization and either decline it
or send it to review. The score ranks transactions for a capacity-limited
review team; the threshold turns the ranking into a decision under the cost
assumptions in ADR 0006. Not intended for: deciding whether a *specific
purchase* was the fraudulent act (the label is partly account-level, ADR
0001), consumer-facing explanations, or any population other than the
provider's (a single e-commerce payment processor, six months of 2017 – 18).

## Training data

Kaggle IEEE-CIS Fraud Detection, labelled training file only: 590,540
transactions over 182 days, 3.50 % positives, identity record present for
24.4 %. Split by time (ADR 0002): train days 1 – 122 (420,066 rows),
validation 123 – 152 (85,044), test 153 – 183 (85,430). Nothing is fit
outside the training window. Details in `docs/eda.md`.

## Evaluation

Validation was used for every decision (feature sets, tuning, threshold,
calibration choice). The test window was scored **once**, at the end.

| metric | validation | test |
|---|---:|---:|
| PR-AUC (primary) | 0.619 | **0.557** |
| ROC-AUC | 0.932 | 0.910 |
| precision / recall / F1 at 0.08 | 0.335 / 0.730 / 0.459 | 0.275 / 0.717 / 0.397 |
| confusion at 0.08 (tp / fp / fn / tn) | 2,106 / 4,185 / 778 / 77,975 | 2,148 / 5,678 / 846 / 76,758 |
| recall at precision ≥ 0.90 | 0.329 | 0.256 |
| precision / recall reviewing top 100 per day | 0.552 / 0.574 | 0.521 / 0.522 |
| precision / recall reviewing top 500 per day | 0.165 / 0.860 | 0.163 / 0.816 |
| Brier (prior: 0.033) / ECE | 0.0191 / 0.0049 | 0.0217 / 0.0067 |
| cost at 0.08 / at 0.5 / approve-all | 230k / 342k / 484k | 262k / 358k / 478k |

The previous candidate (E008, without the composite keys) scored 0.616 /
0.553; it was also evaluated once on test, before E016 was accepted.

Baselines on validation: constant 0.034 PR-AUC; logistic regression 0.288
(26 columns) / 0.402 (all raw columns); untuned XGBoost on raw columns
0.570; tuned without composite keys 0.616.

**Reading the test column.** PR-AUC falls 0.06 from validation to test. The
test month is one month further from the training window, and every
experiment shows the data drifts (`docs/eda.md` §5). Recall at the threshold
holds (0.73 → 0.72) while precision drops (0.33 → 0.27): the model still
finds the fraud, but the score scale has moved — the calibration error
grows from 0.0046 to 0.0060 in the same direction. The chosen policy still
cuts cost by 45 % against approving everything (52 % on validation). A
deployed version would retrain on a schedule and re-select the threshold;
the drift rate here says monthly.

## What the model relies on (`docs/ablation.md`)

The provider's `C*` counts and the `card*` fields are irreplaceable (−0.05
PR-AUC each when removed). The 339 `V` columns take 76 % of split gain but
are almost fully substitutable (−0.005 when removed). `has_identity` is
unused; hour of day and reconstructed card-history features added nothing
over the provider's columns (E005, E007, E017); row-local missing counts,
amount structure and e-mail families added nothing (E013 – E015). The
ablation was run on E008; the shipped model differs from it only by four
frequency-encoded composite keys.

## Limitations and known risks

- **Label semantics.** Positives include routine purchases on an account
  that was later reported. A "false positive" on such an account is not a
  model error under this label; error analysis must not treat it as one.
- **Label maturity.** Training labels are fully mature; production labels
  for the most recent weeks would not be. This makes the offline numbers
  slightly optimistic in a time-invariant way (ADR 0002).
- **Provider features as point-in-time.** `C*`, `D*`, `M*`, `V*`, `id_*` are
  accepted as available at authorization on the host's statement; this
  cannot be verified from anonymised data (`docs/leakage_audit.md`).
- **Drift.** One extra month costs 0.06 PR-AUC and 0.06 precision at the
  threshold. The frequency tables and the calibration map are frozen to the
  training population and age with it.
- **Cost model is assumed.** FN = amount + 15, FP = 0.10 · amount + 2. The
  threshold moves between 0.055 and 0.165 under ±50 % changes
  (`docs/threshold.md`); the direction is robust, the value is a business
  input.
- **Recall-heavy policy.** At 0.08 about 7 % of transactions are declined and
  two thirds of those are legitimate. With a review queue instead of a hard
  decline the false-positive cost falls and the threshold would move lower
  still; with a friction-sensitive merchant it moves higher.
- **Fairness.** No demographic attributes exist in the data; disparate impact
  across, e.g., email domain or device type has not been assessed and would
  need an argument before this model made autonomous decline decisions.
- **Serving is stateless** on purpose: every input is in the request. Any
  future history-based feature (ADR 0004) requires a feature store and a
  parity test before it enters the served model.

## Reproduce

```bash
uv run python scripts/download_data.py
uv run python scripts/train.py --model configs/model/xgboost_f5_interactions.yaml
uv run python scripts/calibrate.py --model-config configs/model/xgboost_f5_interactions.yaml
uv run python scripts/select_threshold.py --run-name xgb_f5_interactions
uv run python scripts/ablation.py --model-config configs/model/xgboost_v2_tuned.yaml
uv run python scripts/evaluate_test.py --run-name xgb_f5_interactions
```
