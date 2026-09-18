# Model card — fraud risk scoring, `xgb_f5_capacity` + sigmoid calibration

## Model details

- **Type:** gradient-boosted trees (XGBoost 3.4, `hist`), 1,600 trees, depth
  12, η 0.05, min_child_weight 1, subsample 0.8, colsample 0.5, L2 1, L1 1
  (E022, from the one-axis sweeps in `docs/xgboost_progression.md`);
  wrapped with a Platt (sigmoid-on-logit) calibrator. One sklearn object
  (`models/xgb_f5_capacity_calibrated.joblib`, 41 MB) does preprocessing,
  scoring and calibration; the API loads exactly that object and refuses to
  start unless it reproduces a frozen 50-row golden (`fraud.serve.parity`).
- **Inputs:** the IEEE-CIS transaction record (393 provider columns) plus the
  identity record (40 columns) when one exists. Twelve high-cardinality
  columns and four row-local composite keys (`card1+addr1`,
  `card1+addr1+P_emaildomain`, `card1+card4`, `card1+card6`) are
  frequency-encoded from the training population;
  22 low-cardinality strings are one-hot encoded with `<missing>` as a
  level; missing numerics are left to the trees. `TransactionDT` and
  `TransactionID` are never inputs.
- **Output:** calibrated probability that the transaction is, or belongs to
  an account that becomes, reported as fraud, and **one action** — approve /
  review / block — with its risk level. `/predict/batch` and `/predict/csv`
  apply the rank-based review policy (block ≥ 0.42, review the top-N
  remaining by the analyst budget, default 200, approve the rest); single
  `/predict` uses the fixed bands (approve < 0.062, review < 0.42). The
  review budget is per **transaction day**, released in proportion to the
  day elapsed, and charged with the reviews earlier requests already issued
  for that day (read from the audit trail; `policy.budget_accounting` says
  whether that memory was available). The evaluation threshold 0.08 (ADR
  0006) is used in this card's metrics and is not served.
- **Version:** `xgb_f5_capacity+sigmoid@<sha256 prefix of the artifact>`,
  returned by `/health` and `/predict`. MLflow: `xgb_f5_capacity` in
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
calibration choice). The test window is a **final temporal reporting
window**: no model was selected on it, but three candidates (E008, E016,
E022) and two policy checks were reported on the same later window during
development, so it is **not a strictly blind holdout** and its numbers carry
a small optimistic bias from having been seen (consultation log in ADR
0002).

| metric | validation | test |
|---|---:|---:|
| PR-AUC (primary) | 0.637 | **0.561** |
| ROC-AUC | 0.933 | 0.907 |
| precision / recall / F1 at 0.08 | 0.346 / 0.743 / 0.472 | 0.262 / 0.691 / 0.380 |
| confusion at 0.08 (tp / fp / fn / tn) | 2,142 / 4,053 / 742 / 78,107 | 2,070 / 5,826 / 924 / 76,610 |
| recall at precision ≥ 0.90 | 0.334 | 0.243 |
| precision / recall reviewing top 100 per day | 0.571 / 0.594 | 0.529 / 0.530 |
| precision / recall reviewing top 500 per day | 0.168 / 0.873 | 0.160 / 0.801 |
| Brier (prior: 0.033) / ECE | 0.0185 / 0.0037 | 0.0214 / 0.0059 |
| cost at 0.08 / at 0.5 / approve-all | 218k / 332k / 487k | 275k / 371k / 481k |

Earlier candidates reported on the same window before the next was
accepted: E008 0.616 / 0.553, E016 0.619 / 0.557. E022's +0.017 validation
gain over E016 became +0.004 on test, and its operating-point numbers are
slightly *worse* than E016's (precision 0.262 vs 0.275 at 0.08, cost 275k vs
262k). More capacity memorises the training window harder and transfers a
little worse over two months; the validation window, adjacent to training,
could not see it. The decision rule (validation decides, test reports) was
applied as written. The follow-up — a drift-aware validation horizon — is
now ADR 0008: rolling backtests confirm E022 on every horizon (its edge
halves with distance but never reverses; +0.007 paired at ≥ 30 days), and
the same backtests measure one month of staleness at ~0.1 PR-AUC, which
the retraining lifecycle (`docs/retraining.md`) recovers.

Baselines on validation: constant 0.034 PR-AUC; logistic regression 0.288
(26 columns) / 0.402 (all raw columns); untuned XGBoost on raw columns
0.570; tuned without composite keys 0.616.

**Reading the test column.** It is one month further out than validation
and has been reported on at several milestones (above). PR-AUC falls 0.08
from validation to test. The
test month is one month further from the training window, and every
experiment shows the data drifts (`docs/eda.md` §5). Recall at the threshold
mostly holds (0.74 → 0.69) while precision drops (0.35 → 0.26): the model still
finds the fraud, but the score scale has moved — the calibration error
grows from 0.0046 to 0.0060 in the same direction. The chosen policy still
cuts cost by 43 % against approving everything (55 % on validation). A
deployed version would retrain on a schedule and re-select the threshold;
the drift rate here says monthly.

**By subgroup** (`docs/subgroups.md`, validation, served thresholds held
fixed). The global PR-AUC is a blend of two segments: transactions with an
identity record (18 % of rows, half the fraud) score **0.80** and the
policy reaches 89 % of their fraud at block + review; transactions without
one — product `W`, 82 % of rows, the other half of the fraud — score
**0.46** and the policy reaches 67 %. Calibration holds in every group
(mean score within ~0.01 of prevalence). The 80 % block-precision bar is
missed for amounts under 25 (0.66 on 289 blocks) and for discover cards
(0.67 on 57); block + review recall falls to 56 % above 1,000 (27 fraud).
Within the month, PR-AUC drifts 0.72 → 0.61 → 0.57 across ten-day blocks.

## What the model relies on (`docs/ablation.md`)

Per transaction, `POST /explain` gives the TreeSHAP contributions of the
booster summed per source column and family, with the calibration step
stated separately (`docs/explanation.md`); it explains rank, not fraud.

The provider's `C*` counts and the `card*` fields are irreplaceable (−0.05
PR-AUC each when removed). The 339 `V` columns take 76 % of split gain but
are almost fully substitutable (−0.005 when removed). `has_identity` is
unused; hour of day and reconstructed card-history features added nothing
over the provider's columns (E005, E007, E017); row-local missing counts,
amount structure and e-mail families added nothing (E013 – E015). Family
cuts (`docs/feature_sets.md`): the provider's `C`/`D`/`M`/`V` families are
worth 0.19 PR-AUC; the frequency / key features here add ≈ 0.02; `V` alone
is optional on the shipped set (a 100-input variant scores the same). The
ablation was run on E008; the shipped model differs from it only by four
frequency-encoded composite keys.

## Error analysis (`docs/error_analysis.md`)

Confident false positives are the fraud archetype (new card, product `C`,
no billing address, self-addressed e-mail) performed by legitimate
customers — 27 % of what the policy would block on test, indistinguishable
row by row. Confident false negatives (28 % of test fraud) are two things:
propagated labels on ordinary purchases of reported accounts (71 % share an
entity with other labelled fraud; not recoverable at authorization) and
established cards used for the mainstream product (recoverable only via
per-entity deviation features on that slice). Analysed on E016, re-checked
on E022 with the same profile.

## Limitations and known risks

- **Autonomous blocking is a demonstration policy.** The overall block-precision
  target does not hold in every subgroup. In a real payment system, weak
  subgroups would need additional policy review and likely more conservative
  treatment (`docs/subgroups.md`). Documenting this limitation is sufficient
  for Stage 1; it does not justify introducing subgroup-specific thresholds
  without further evidence and an explicit policy decision.
- **Label semantics.** Positives include routine purchases on an account
  that was later reported. A "false positive" on such an account is not a
  model error under this label; error analysis must not treat it as one.
- **Label maturity.** Training labels are fully mature; production labels
  are not for 120 days (the host's reporting window). The feedback loop
  (`docs/feedback.md`, ADR 0009) ages every scored transaction against
  that window: eventual performance is computed on closed cohorts only,
  because for 90 days after a month ends every arrived label is a
  positive, and the first cohorts to close are the ones nearest the
  training window (their PR-AUC 0.743 vs 0.637 for the full month).
  Retraining on matured labels adds the window to the model's staleness.
- **Provider features as point-in-time.** `C*`, `D*`, `M*`, `V*`, `id_*` are
  accepted as available at authorization on the host's statement; this
  cannot be verified from anonymised data (`docs/leakage_audit.md`).
- **Drift.** One extra month costs 0.08 PR-AUC and 0.08 precision at the
  threshold, and the higher-capacity model loses more of its validation
  advantage than its predecessor did. `docs/monitoring.md` describes the
  runtime monitor (score / action / feature PSI; eventual performance with
  matured labels) that is meant to catch this in production; it reproduced
  the drop on a reporting-window day (eventual PR-AUC 0.617 vs 0.637). The frequency tables and the calibration map are frozen to the
  training population and age with it.
- **Cost model is assumed.** FN = amount + 15, FP = 0.10 · amount + 2. The
  threshold moves between 0.055 and 0.165 under ±50 % changes
  (`docs/threshold.md`); the direction is robust, the value is a business
  input.
- **Recall-heavy single threshold.** At 0.08 about 7 % of transactions are
  declined and two thirds of those are legitimate. `docs/review_policy.md`
  replaces it with block ≥ 0.42 / review by daily rank / approve, which
  costs 30 % less on validation; on test the block precision drifts from
  0.80 to 0.71 and a fixed review threshold overshoots its budget by 25 %,
  so reviewing should be rank-based.
- **Fairness.** No demographic attributes exist in the data, so no fairness
  claim is made. Subgroup robustness by product, card network and type,
  identity presence, e-mail provider family, device class, amount band and
  time is assessed (`docs/subgroups.md`): no group is scored above its own
  fraud rate, and the weak segment (no identity record) is a coverage gap,
  not an over-flagging one. Autonomous decline decisions would still need an
  argument about who bears the false blocks that these groupings cannot
  make.
- **The features are stateless; the service is not.** Every model input
  is in the request, so the *score* of a transaction depends on nothing but
  its own row (the startup parity check covers exactly this), and any
  future history-based feature (ADR 0004) requires a feature store and a
  parity test before it enters the served model. The *action* under the
  rank policy is another matter: the service keeps an append-only audit
  trail (SQLite: every scored transaction with its inputs, model version,
  policy and result; the request log; delayed labels posted to
  `/outcomes`) and reads it at decision time to charge a day's review
  budget with reviews already issued by earlier requests. Two servers with
  different audit histories can therefore assign different actions to the
  same batch, and replaying a day reproduces its actions only in the
  original order; with auditing disabled the budget is per request and the
  memory is gone. The audit trail is also what an investigation, the drift
  monitor and the feedback loop read.

## Reproduce

```bash
uv run python scripts/download_data.py
uv run python scripts/train.py --model configs/model/xgboost_f5_capacity.yaml
uv run python scripts/calibrate.py --model-config configs/model/xgboost_f5_capacity.yaml
uv run python scripts/select_threshold.py --run-name xgb_f5_capacity
uv run python scripts/freeze_artifact.py --run-name xgb_f5_capacity
uv run python scripts/ablation.py --model-config configs/model/xgboost_v2_tuned.yaml
uv run python scripts/evaluate_test.py --run-name xgb_f5_capacity
```
