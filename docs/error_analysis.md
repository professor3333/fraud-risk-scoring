# Final evaluation and error analysis

Analysed on the shipped model, E022 (feature set `f5_interactions`,
depth 12 × 1,600 trees, sigmoid calibration, threshold 0.08, review bands
0.062 / 0.42; `reports/final/` is its bundle). It was first written on the
previous shipped model, E016; the re-analysis on E022 gave the same
profile with slightly different counts, and every number below is now
E022's. The test window (days 153 – 183, 85,430 transactions, 2,994
fraud) is the final temporal reporting window — consulted at several milestones (ADR 0002's log), never
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
| PR-AUC / ROC-AUC | **0.561** / 0.907 |
| at 0.08: precision / recall / F1 | 0.262 / 0.691 / 0.380 |
| confusion at 0.08 (tp / fp / fn / tn) | 2,070 / 5,826 / 924 / 76,610 |
| recall at precision ≥ 0.90 | 0.243 |
| reviewing top 100 / 200 / 500 per day: recall | 0.53 / 0.66 / 0.80 |
| Brier (prior 0.034) / ECE | 0.0214 / 0.0059 |
| policy: block ≥ 0.42 | 60 / day, 73 % fraud, catches 44 % of fraud |
| policy: review 0.062 – 0.42 | 252 / day, 11 % fraud, recall block + review 0.72 |
| policy: approve < 0.062 | 2,536 / day, 1.1 % fraud remaining |
| cost at 0.08 / at 0.5 / approve-all | 275k / 371k / 481k |

Validation → test: PR-AUC 0.637 → 0.561, recall at 0.08 holds (0.74 →
0.69), precision drops (0.35 → 0.26), calibration loosens (ECE 0.004 →
0.006). One month of drift, as the model card says.

## 2. The three groups inspected

Categories from `error_analysis.csv` (bands from the review policy):

| category | rows | meaning |
|---|---:|---|
| confident true fraud | 1,307 | p ≥ 0.42 and fraud |
| **high-confidence false positive** | 490 | p ≥ 0.42 and legitimate — 27 % of everything the policy would block |
| **high-confidence false negative** | 844 | p < 0.062 and fraud — 28 % of all test fraud, scored *safer than average* (mean p 0.028 vs 0.041) |
| review band | 7,550 | 0.062 ≤ p < 0.42 |
| confident true legit | 75,239 | |

Profiles (share of rows unless stated):

| | all test | confident fraud | **HC false positive** | **HC false negative** |
|---|---:|---:|---:|---:|
| product `C` | 0.11 | 0.69 | **0.62** | 0.16 |
| product `W` | 0.78 | 0.12 | 0.21 | **0.71** |
| identity record present | 0.21 | 0.87 | **0.77** | 0.28 |
| recipient e-mail present | 0.21 | 0.87 | **0.77** | 0.28 |
| billing `addr1` missing | 0.11 | 0.68 | **0.62** | 0.16 |
| `D1 = 0` (card first seen today) | 0.42 | 0.67 | **0.67** | 0.48 |
| `M4 = M2` | 0.10 | 0.57 | **0.53** | 0.16 |
| credit card | 0.24 | 0.50 | **0.63** | 0.34 |
| mobile device | 0.09 | 0.43 | **0.40** | 0.16 |
| median amount | 68 | 48 | 54 | 75 |
| `card1` seen in training | 0.99 | 0.98 | 0.97 | 0.99 |
| shares `card1+addr1` with ≥ 2 other test frauds | — | 0.17 | — | **0.61** |
| median fraud share of that entity's test rows | — | — | — | **0.19** |

## 3. What fools the model

### High-confidence false positives are the fraud archetype, done by real customers

The 490 legitimate transactions the model is surest about look almost
exactly like confident fraud: product `C`, a card first seen today, no
billing address, an identity record, the purchaser's own e-mail as the
recipient, `M4 = M2`, a credit card, often on mobile. The six highest-scored
legitimate rows (p 0.992 – 0.995) are all product `C`, `addr1` missing,
`P_emaildomain = R_emaildomain` (gmail), `M4 = M2`, `D1 ≤ 2`, `C1` 5 – 13.

The model has learned "new card + product `C` + no address + self-addressed
e-mail" as fraud, and it is right 73 % of the time on test. The other 27 %
are, by every field the model can see, indistinguishable: most plausibly
first-time customers buying a digital / gift-type product for themselves.
**Nothing in the row separates them; only history or the outcome would.**
This is the strongest argument for the review band: a transaction in this
archetype should be *reviewed* (a 3-dollar delay), not blocked outright (a
lost customer nearly three times in ten).

Two details worth a note. `C1` is high (5 – 13) for these rows — the
provider's count of something linked to the card is elevated on both the
true and the false positives, so it is part of the archetype, not a way
out of it. And the false-positive rate of the block band drifted from 20 %
on validation to 27 % on test, so the archetype's fraud share is falling
over time — the block threshold must be re-set at every retrain.

### High-confidence false negatives are fraud that looks like everyone else

The 844 frauds the model is surest are safe look like the population, not
like fraud: 71 % product `W` (the mainstream product, 2 % fraud overall),
billing address present, no recipient e-mail, usually no identity record,
established cards. The lowest-scored frauds (p < 0.0014) include a `W`
purchase on a card with 139 linked entities and 422 prior transactions
(`C1`, `C13`) at a 76-day-old address, and four 58 – 108 dollar `W` debit
purchases over two weeks from one yahoo address at the same billing
address, `D1` 13 – 28 days.

Two mechanisms, and the entity check separates them:

1. **Label propagation.** 61 % of these frauds share their `card1+addr1`
   entity with at least two other test frauds (confident fraud: 17 %), yet
   the median entity is only 19 % fraud. That is the signature ADR 0001
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

69 % product `C`, 87 % with identity, 68 % no billing address, 87 % with a
recipient e-mail, 67 % on a card first seen today, median amount 48. The
model is excellent at this pattern (the top 25 reviews per day are 90 %
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
