# Final evaluation and error analysis

Analysed on the model shipped at the time (E016: feature set
`f5_interactions`, E008 parameters, sigmoid calibration, threshold 0.08,
review bands 0.067 / 0.42) and re-checked on the current shipped model
(E022, same features, depth 12 × 1,600 trees; `reports/final/` now holds
E022's bundle). The re-check gave the same profile: 490 high-confidence false
positives (62 % product `C`, 62 % no address, 67 % new card), 844
high-confidence false negatives (71 % product `W`, 71 % sharing an entity
with ≥ 2 other test frauds), 1,307 confident true frauds. The numbers below
are E016's; the reading is unchanged. The test window (days
153 – 183, 85,430 transactions, 2,994 fraud) is the final temporal
reporting window — consulted at several milestones (ADR 0002's log), never
used to select a model; `reports/final/` is the same set of predictions as
the candidate's `evaluate_test` run, laid out for inspection. Nothing in
this document changes a decision.

`reports/final/`: `metrics.json` · `precision_recall_curve.png` ·
`roc_curve.png` · `confusion_matrix.png` · `threshold_analysis.csv` ·
`feature_importance.{csv,png}` · `error_analysis.csv` (every test row with
its probability, action and error category).

## 1. Headline numbers (test)

| | |
|---|---|
| PR-AUC / ROC-AUC | **0.557** / 0.910 |
| at 0.08: precision / recall / F1 | 0.275 / 0.717 / 0.397 |
| confusion at 0.08 (tp / fp / fn / tn) | 2,148 / 5,678 / 846 / 76,758 |
| recall at precision ≥ 0.90 | 0.256 |
| reviewing top 100 / 200 / 500 per day: recall | 0.52 / 0.66 / 0.82 |
| Brier (prior 0.034) / ECE | 0.0217 / 0.0067 |
| policy: block ≥ 0.42 | 60 / day, 71 % fraud, catches 43 % of fraud |
| policy: review 0.067 – 0.42 | 248 / day, 13 % fraud, recall block + review 0.75 |
| policy: approve < 0.067 | 2,540 / day, 1.0 % fraud remaining |
| cost at 0.08 / at 0.5 / approve-all | 262k / 358k / 478k |

Validation → test: PR-AUC 0.619 → 0.557, recall at 0.08 holds (0.73 →
0.72), precision drops (0.33 → 0.27), calibration loosens (ECE 0.005 →
0.007). One month of drift, as the model card says.

## 2. The three groups inspected

Categories from `error_analysis.csv` (bands from the review policy):

| category | rows | meaning |
|---|---:|---|
| confident true fraud | 1,277 | p ≥ 0.42 and fraud |
| **high-confidence false positive** | 522 | p ≥ 0.42 and legitimate — 29 % of everything the policy would block |
| **high-confidence false negative** | 762 | p < 0.067 and fraud — 25 % of all test fraud, scored *safer than average* (mean p 0.028 vs 0.042) |
| review band | 7,435 | 0.067 ≤ p < 0.42 |
| confident true legit | 75,434 | |

Profiles (share of rows unless stated):

| | all test | confident fraud | **HC false positive** | **HC false negative** |
|---|---:|---:|---:|---:|
| product `C` | 0.11 | 0.70 | **0.57** | 0.18 |
| product `W` | 0.78 | 0.10 | 0.22 | **0.67** |
| identity record present | 0.21 | 0.89 | **0.77** | 0.32 |
| recipient e-mail present | 0.21 | 0.88 | **0.76** | 0.32 |
| billing `addr1` missing | 0.11 | 0.69 | **0.57** | 0.18 |
| `D1 = 0` (card first seen today) | 0.42 | 0.68 | **0.62** | 0.48 |
| `M4 = M2` | 0.10 | 0.58 | **0.49** | 0.18 |
| credit card | 0.24 | 0.53 | **0.62** | 0.32 |
| mobile device | 0.09 | 0.44 | **0.38** | 0.19 |
| median amount | 68 | 48 | 71 | 75 |
| `card1` seen in training | 0.99 | 0.98 | 0.97 | 0.99 |
| shares `card1+addr1` with ≥ 2 test frauds | — | 0.21 | — | **0.68** |
| median fraud share of that entity's test rows | — | — | — | **0.17** |

## 3. What fools the model

### High-confidence false positives are the fraud archetype, done by real customers

The 522 legitimate transactions the model is surest about look almost
exactly like confident fraud: product `C`, a card first seen today, no
billing address, an identity record, the purchaser's own e-mail as the
recipient, `M4 = M2`, a credit card, often on mobile. The six highest-scored
legitimate rows (p 0.992 – 0.995) are all product `C`, `addr1` missing,
`P_emaildomain = R_emaildomain` (gmail / hotmail), `M4 = M2`, `D1 ≤ 6`,
`C1` 8 – 14.

The model has learned "new card + product `C` + no address + self-addressed
e-mail" as fraud, and it is right 71 % of the time on test. The other 29 %
are, by every field the model can see, indistinguishable: most plausibly
first-time customers buying a digital / gift-type product for themselves.
**Nothing in the row separates them; only history or the outcome would.**
This is the strongest argument for the review band: a transaction in this
archetype should be *reviewed* (a 3-dollar delay), not blocked outright (a
lost customer three times in ten).

Two details worth a note. `C1` is high (8 – 14) for these rows — the
provider's count of something linked to the card is elevated on both the
true and the false positives, so it is part of the archetype, not a way
out of it. And the false-positive rate of the block band drifted from 20 %
on validation to 29 % on test, so the archetype's fraud share is falling
over time — the block threshold must be re-set at every retrain.

### High-confidence false negatives are fraud that looks like everyone else

The 762 frauds the model is surest are safe look like the population, not
like fraud: 67 % product `W` (the mainstream product, 2 % fraud overall),
billing address present, no recipient e-mail, usually no identity record,
established cards. The lowest-scored frauds (p < 0.0015) include a `W`
purchase on a card with 139 linked entities and 422 prior transactions
(`C1`, `C13`) at a 76-day-old address, and three 10 – 20 dollar product-`S`
purchases spread over two weeks on a 170 – 186-day-old card with the same
address and iCloud e-mail.

Two mechanisms, and the entity check separates them:

1. **Label propagation.** 68 % of these frauds share their `card1+addr1`
   entity with at least two other test frauds (confident fraud: 21 %), yet
   the median entity is only 17 % fraud. That is the signature ADR 0001
   warned about: an account is reported, its *subsequent* transactions
   are labelled fraud, and many of those are ordinary purchases. The model
   cannot see "this account was reported last week" — that information
   arrives after the fact — so under this label definition these rows are
   *not recoverable at authorization*. They are a property of the label,
   not a failure of the model.
2. **Fraud on established accounts** (the remainder, ~⅓): a stolen card
   with a long, clean history used for the mainstream product with the
   right address. The row carries no archetype signal. What *could* catch
   it is deviation from the card's own habit — amount, product, time-of-day
   versus its history — which is exactly what E007 / E017 built and found
   no *aggregate* gain from. This slice says why the aggregate was flat:
   the deviation signal lives in ~⅓ of ¼ of the fraud, and the provider's
   `C` / `D` columns already carry the coarse version. A targeted
   experiment — history features evaluated on the `W`-product,
   established-card slice only — is the follow-up this analysis suggests,
   and the one this project has not run.

### Confident true fraud is one pattern

70 % product `C`, 89 % with identity, 69 % no billing address, 88 % with a
recipient e-mail, 68 % on a card first seen today, median amount 48. The
model is excellent at this pattern (the top 25 reviews per day are 89 %
fraud on validation) and the pattern is a large share of the labelled
fraud. It is also the pattern the false positives share.

## 4. What this says about the system

- The model's precision problem is one archetype: new-card, no-address,
  self-addressed product `C`. The policy answer is review, not block; the
  modelling answer would need information the row does not contain.
- The model's recall problem is two things: propagated labels (not
  recoverable at authorization; a reporting-latency issue) and established-
  account takeover (recoverable only through per-entity deviation
  features, which the aggregate experiments could not justify but this
  slice motivates).
- Both problems are product-shaped. A per-product operating threshold — or
  a per-product model — is the obvious next structural experiment.
- `V` still carries 74 % of split gain (`feature_importance.png`, `V258`
  alone 12 %), consistent with `docs/feature_sets.md`: it is the model's
  preferred encoding of the archetype, and it is replaceable.
