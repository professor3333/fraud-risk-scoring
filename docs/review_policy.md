# Threshold analysis and the review policy

Model: `xgb_f5_interactions` + sigmoid calibration. Evidence:
`reports/policy/` from `uv run python scripts/review_policy.py --run-name
xgb_f5_interactions --with-test`. Everything is *chosen* on the validation
window (85,044 transactions over 30 days, 96 fraud per day on average) and
*checked* on the final temporal reporting window (which has been consulted
at several milestones — ADR 0002's log).

## 1. Operating points

`xgb_f5_interactions_operating_points.csv` tabulates precision, recall, F1,
TP / FP / TN / FN, the number flagged and the fraud caught per day at every
threshold from 0.01 to 0.99 (fig. left). The curve says what a single
threshold can and cannot do: at 0.5 the model is right 84 % of the time but
finds 41 % of fraud; at 0.08 (ADR 0006) it finds 73 % and is right 33 % of
the time. No single number does both.

## 2. Three actions instead of one threshold

The model outputs a probability; the business has three actions, with
different costs:

| action | cost of the action | when wrong |
|---|---|---|
| block | none | legitimate customer declined: 0.10 · amount + 2 (ADR 0006) |
| review | analyst handling 3 per transaction; 0.02 · amount delay friction if legitimate | fraud is assumed caught in review |
| approve | none | fraud approved: amount + 15 |

Two cut-points define the bands, both read from validation:

- **Block when the calibrated probability ≥ 0.42.** Chosen as the lowest
  threshold at which blocked transactions are at least 80 % fraud
  (`block_min_precision` in `configs/policy.yaml`). Explainable to a
  customer-facing team: "we only decline outright when four in five such
  transactions are fraud". On validation: 53 blocks per day, 42 of them
  fraud, 11 legitimate customers declined per day; 44 % of all fraud.
- **Review down to the score that fills the analyst budget.** With *N*
  reviews per day, the review threshold is the score at which the band
  [review, 0.42) holds *N* transactions per day on validation. The
  remainder is approved.

Policy table on validation (one row per budget):

| reviews / day | review threshold | reviewed fraud / day | review precision | recall block + review | fraud approved / day | total cost |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.249 | 8.8 | 0.35 | 0.53 | 44.8 | 261,552 |
| 50 | 0.176 | 14.3 | 0.29 | 0.59 | 39.4 | 231,466 |
| 100 | 0.113 | 21.9 | 0.22 | 0.67 | 31.8 | 200,722 |
| **200** | **0.067** | **31.3** | 0.16 | **0.77** | 22.4 | **160,748** |
| 500 | 0.030 | 41.2 | 0.08 | 0.87 | 12.4 | 161,903 |
| 1000 | 0.015 | 48.3 | 0.05 | 0.94 | 5.4 | 214,745 |

Reading:

- Adding a review action **cuts validation cost from 230k (decline-only at
  0.08, `docs/threshold.md`) to 161k at 200 reviews/day** — 30 % lower,
  and 67 % below approving everything — because a review costs 3 and a
  decline costs a customer.
- Cost is flat between 200 and 500 reviews/day and rises again at 1,000:
  beyond ~500 the marginal review catches 0.5 % fraud and costs more than
  it saves. The cost-optimal budget under these assumptions is **200 –
  500 analysts-reviews per day** for ~2,850 daily transactions.
- The bands for a 200/day team, in calibrated probability:
  **approve < 0.067 · review 0.067 – 0.42 · block ≥ 0.42.** These are
  data-derived, not round numbers, and they move with the model.

## 3. Review budget as a pure ranking problem

If the only lever is "which *N* transactions do analysts look at", the
question is ranking, not thresholding (fig. right):

| reviews / day | precision @ budget | recall @ budget | fraud caught / day | implied score cut |
|---:|---:|---:|---:|---:|
| 25 | 0.89 | 0.23 | 22.3 | 0.78 |
| 50 | 0.78 | 0.40 | 38.8 | 0.47 |
| 100 | 0.55 | 0.57 | 55.2 | 0.20 |
| 200 | 0.35 | 0.72 | 69.7 | 0.085 |
| 500 | 0.17 | 0.86 | 82.7 | 0.034 |
| 1000 | 0.09 | 0.94 | 90.1 | 0.016 |

A 100-analyst-review day catches 55 of the ~96 daily frauds at 55 %
precision; 500 reviews catch 83. The first 25 reviews are 89 % fraud — the
model's top of the ranking is very clean.

## 4. What the reporting window says

Same thresholds, one month later:

| reviews / day (validation sizing) | actually reviewed / day | block precision | recall block + review | total cost |
|---:|---:|---:|---:|---:|
| 100 | 128 | 0.71 | 0.65 | 222,945 |
| 200 | 250 | 0.71 | 0.75 | 186,075 |
| 500 | 574 | 0.71 | 0.85 | 183,676 |

Two things drift, and both matter operationally:

1. **Block precision falls from 0.80 to 0.71.** The "four in five" promise
   made on validation is "seven in ten" a month later. Probabilities drift
   upward with the base rate (Brier / ECE also worsen, model card), so a
   fixed 0.42 blocks more legitimate customers than intended.
2. **A fixed review threshold overshoots the budget by 25 %** (250
   reviewed at the "200" setting) for the same reason. A rank-based rule —
   "review the top *N* below the block line each day" — holds the volume
   by construction and degrades gracefully; a threshold rule needs
   re-sizing whenever the score distribution moves.

Recommendation recorded for deployment: block by threshold (re-selected at
each retrain against the precision bar), review by **daily rank with a
fixed budget**, approve the rest. Recall at the 200-review budget is 0.75
on test versus 0.77 on validation — the ranking holds up better than the
probability scale.

## 5. What this is not

The costs are assumptions (ADR 0006), the review is assumed to catch every
fraud it sees, and analyst capacity is treated as free below the budget.
Any of these would move the optimal budget; the *shape* — a clean top of the
ranking, a flat cost floor across a wide band of budgets, and drift that
hurts thresholds more than ranks — is the durable finding.

## Re-check for the current shipped model (`xgb_f5_capacity`, E022)

`reports/policy/xgb_f5_capacity_*`. Block threshold at the 80 % precision bar
is again **0.42** (56 blocks/day, 45 fraud). For a 200-review/day team the
review threshold is 0.062; validation recall block + review **0.776**, cost
157k (E016: 0.767, 161k). On test: recall 0.727 (E016 0.746), block
precision 0.727 (0.710), 273 reviewed/day at the "200" sizing (E016: 250).
The same two drifts as before, slightly larger on the review volume; the
rank-based review recommendation stands.

## Implemented in the service

`POST /predict/batch` and `POST /predict/csv` apply this policy by default
(`policy: "rank"`, `review_budget` per request, server default 200): block
at `bands.block`, review the highest-scored remaining rows up to the budget,
approve the rest, and report the lowest probability actually reviewed
(`policy.review_cutoff`). Live check, budget 200: validation day 130 →
block 47 / review **200** / approve 2,560; test day 170 → block 54 / review
**200** / approve 2,155. The fixed-threshold policy on the same two days
reviews 141 and 222. `fraud.evaluate.policy.apply_rank_policy` is the
function; `policy: "threshold"` keeps the fixed bands for comparison.

The service returns **one** command per transaction (`action`) with its
`risk_level`; the single-threshold `decision` at 0.08 was removed from the
API (v0.4.1) because a response carrying both `decision: decline` and
`action: review` gave the payment system two competing instructions. The
0.08 threshold remains the evaluation operating point in the reports.
