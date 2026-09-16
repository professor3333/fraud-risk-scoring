# Per-prediction explanation

`POST /explain` answers "why does transaction 123 score 0.82?" for one
transaction; the dashboard's row inspector shows the same answer under
*Why this score*. It exists next to the global importance work
(`docs/ablation.md`: gain, group permutation, ablation) because an analyst
looking at one row needs the row's own reasons, not the model's average
ones.

## How

XGBoost computes exact TreeSHAP contributions for its own trees
(`booster.predict(..., pred_contribs=True)`): one number per model input
per row, in log-odds, that sum with the bias to the raw margin. That is the
SHAP number without the `shap` package — no new dependency, nothing added
to the image. `fraud.evaluate.explain` adds the two things the raw output
lacks:

1. **The analyst's vocabulary.** The model has 491 inputs: one column per
   one-hot level (`cat__ProductCD_C`, `cat__ProductCD_W`, …), one per
   frequency table (`freq__freq_card1`), one per raw numeric. Contributions
   are summed per *source* — `ProductCD`, `card1 frequency`, `C1` — and
   per family (the ablation groups), and the row's own value is printed
   beside each. Frequency encodings stay separate from the raw column they
   encode (`card1` vs `card1 frequency`), because they are different
   evidence: the card's identity versus how common it was in training.
2. **The calibration layer, stated.** The contributions explain the
   pipeline's *raw* score. The served probability is that score after the
   sigmoid map of ADR 0007. The response carries both, plus the arithmetic
   (`bias + shown + other = raw_margin → raw_probability → fraud_probability`),
   and never pretends the pieces sum to the calibrated number.

A test asserts the contributions reproduce the raw score to float32
precision on every fixture row, that every model input maps to exactly one
source and family, that the endpoint returns the same probability as
`/predict` for the same row, and that explaining is not audited (it
decides nothing).

## An example

The highest-ranked row of the synthetic dashboard sample, budget 25:

```
raw margin −0.65 = bias −2.90 + C1=14 (+0.96) + C11=11 (+0.53) + C13=1 (+0.47)
                 + TransactionAmt=31.8 (−0.46) + C4=10 (+0.41) + card3 frequency=0 (−0.36) + other
→ raw 0.34 → calibrated 0.54 → block
by family: C +2.69 · V +1.11 · frequency −0.66 · amount −0.46 · card −0.30
```

Read: the provider's `C` counts carry this row (several linked-entity
counts are high for a card seen once), the amount and the card's rarity
pull the other way, and calibration lifts 0.34 to 0.54 because the raw
score under-states risk in this range (ADR 0007's reliability diagram).

## What it is not

- **Not a fraud verdict.** Under ADR 0001 a positive label includes routine
  purchases on an account that was later reported; the same signals appear
  on a legitimate purchase that merely resembles the archetype
  (`docs/error_analysis.md`: 27 % of blocks on test). The endpoint's `note`
  and the dashboard say so: this is why the transaction *ranks* where it
  does.
- **Not causal.** A contribution is the model's attribution given its other
  inputs; `C1 = 14` "raises the score" in this row, not in general.
- **Not a global statement.** For what the model relies on across the
  population, `docs/ablation.md`; for where it fails, `docs/subgroups.md`.

## API

```bash
curl -X POST http://127.0.0.1:8000/explain?top_k=8 -H 'content-type: application/json' \
  -d '{"TransactionID": 2987133, "TransactionDT": 86506, "TransactionAmt": 31.8, "ProductCD": "C", ...}'
```

Same request body and validation as `/predict`; `top_k` 1 – 50 (default
8). Response: `transaction_id`, `model_version`, `fraud_probability`
(served), `raw_probability`, `raw_margin`, `bias`, `signals`
(`feature`, `value`, `contribution`, `family`), `other_contribution`,
`families`, `note`.
