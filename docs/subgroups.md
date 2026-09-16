# Subgroup robustness

The model card reported one validation PR-AUC (0.637) and said that
performance across e-mail domain or device type had not been assessed. This
is that assessment. It is **not a fairness analysis** — the data carries no
demographic attributes, and nothing here is a proxy claim about people. It
asks whether the global number hides a group on which the model, or the
served policy, fails.

`uv run python scripts/subgroups.py` → `reports/subgroups/validation.csv`,
`findings.md`. Champion E022 on the validation window (days 123 – 152,
85,044 rows, 2,884 fraud), the served policy's thresholds held fixed
(block ≥ 0.42, review ≥ 0.062): per level, *n*, share of the window's
fraud, prevalence, mean score, PR-AUC, block precision, recall at block
and at block + review, flag rate. PR-AUC is suppressed under 30 positives;
levels under 500 rows are pooled as `<small>`.

## The table

| grouping | level | n | share of fraud | prevalence | mean score | PR-AUC | Δ vs all | block precision | recall block | recall block+review | flag rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | | 85,044 | 100 % | 3.4 % | 0.036 | **0.637** | | 0.80 | 0.47 | 0.78 | 9.0 % |
| product | C | 8,490 | 37 % | 12.6 % | 0.132 | 0.804 | +0.17 | 0.79 | 0.71 | 0.89 | 27 % |
| | H | 2,223 | 5 % | 6.4 % | 0.071 | 0.756 | +0.12 | 0.80 | 0.65 | 0.87 | 18 % |
| | R | 3,219 | 5 % | 4.3 % | 0.050 | 0.844 | +0.21 | 0.87 | 0.65 | 0.93 | 11 % |
| | S | 1,412 | 4 % | 7.7 % | 0.051 | 0.764 | +0.13 | 0.98 | 0.41 | 0.81 | 14 % |
| | **W** | 69,700 | 49 % | 2.0 % | 0.022 | **0.457** | **−0.18** | 0.80 | 0.25 | **0.66** | 6 % |
| card network | american express | 740 | 1 % | 4.2 % | 0.050 | 0.836 | +0.20 | 0.96 | 0.68 | 0.87 | 12 % |
| | discover | 936 | 4 % | 12.8 % | 0.092 | 0.608 | −0.03 | **0.67** | 0.32 | 0.90 | 33 % |
| | mastercard | 28,041 | 30 % | 3.1 % | 0.034 | 0.643 | +0.01 | 0.78 | 0.46 | 0.79 | 9 % |
| | visa | 55,323 | 65 % | 3.4 % | 0.036 | 0.630 | −0.01 | 0.81 | 0.48 | 0.76 | 9 % |
| card type | credit | 18,155 | 47 % | 7.5 % | 0.073 | 0.701 | +0.06 | 0.78 | 0.52 | 0.85 | 19 % |
| | debit | 66,887 | 53 % | 2.3 % | 0.026 | 0.578 | −0.06 | 0.82 | 0.43 | 0.71 | 6 % |
| identity | present | 14,750 | 49 % | 9.7 % | 0.099 | 0.801 | +0.16 | 0.81 | 0.68 | 0.89 | 21 % |
| | **absent** | 70,294 | 51 % | 2.1 % | 0.022 | **0.457** | **−0.18** | 0.78 | 0.26 | **0.67** | 7 % |
| e-mail family | gmail | 33,569 | 50 % | 4.3 % | 0.045 | 0.677 | +0.04 | 0.80 | 0.53 | 0.78 | 11 % |
| | yahoo | 15,623 | 10 % | 1.9 % | 0.023 | 0.530 | −0.11 | 0.83 | 0.39 | 0.74 | 7 % |
| | missing | 13,844 | 14 % | 2.9 % | 0.027 | 0.572 | −0.06 | 0.86 | 0.33 | 0.74 | 8 % |
| | microsoft | 7,861 | 12 % | 4.3 % | 0.050 | 0.650 | +0.01 | 0.71 | 0.55 | 0.80 | 12 % |
| | anonymous | 4,824 | 3 % | 1.9 % | 0.026 | 0.525 | −0.11 | 0.74 | 0.40 | 0.74 | 7 % |
| | aol | 4,172 | 4 % | 2.9 % | 0.026 | 0.689 | +0.05 | 0.96 | 0.44 | 0.70 | 5 % |
| | other | 3,999 | 6 % | 4.3 % | 0.034 | 0.755 | +0.12 | 0.90 | 0.35 | 0.94 | 10 % |
| | apple | 1,152 | 1 % | 3.3 % | 0.035 | 0.497 | −0.14 | 0.77 | 0.34 | 0.71 | 11 % |
| device | missing | 70,694 | 52 % | 2.1 % | 0.023 | 0.461 | −0.18 | 0.78 | 0.26 | 0.67 | 7 % |
| | desktop | 8,398 | 27 % | 9.4 % | 0.100 | 0.815 | +0.18 | 0.81 | 0.72 | 0.90 | 21 % |
| | mobile | 5,952 | 21 % | 10.2 % | 0.101 | 0.790 | +0.15 | 0.81 | 0.64 | 0.89 | 23 % |
| amount | 0 – 25 | 6,096 | 12 % | 5.6 % | 0.064 | 0.625 | −0.01 | **0.66** | 0.56 | 0.79 | 14 % |
| | 25 – 50 | 21,342 | 23 % | 3.1 % | 0.033 | 0.733 | +0.10 | 0.85 | 0.58 | 0.81 | 7 % |
| | 50 – 100 | 23,198 | 25 % | 3.1 % | 0.031 | 0.674 | +0.04 | 0.88 | 0.51 | 0.75 | 7 % |
| | 100 – 250 | 25,074 | 26 % | 2.9 % | 0.032 | 0.567 | −0.07 | 0.80 | 0.39 | 0.74 | 9 % |
| | 250 – 1000 | 8,207 | 14 % | 5.0 % | 0.048 | 0.569 | −0.07 | 0.76 | 0.31 | 0.82 | 16 % |
| | > 1000 | 1,127 | 1 % | 2.4 % | 0.039 | (27 fraud) | | 0.50 | 0.22 | 0.56 | 14 % |
| ten-day block | 123 – 132 | 30,681 | 37 % | 3.4 % | 0.038 | 0.718 | +0.08 | 0.83 | 0.55 | 0.82 | 9 % |
| | 133 – 142 | 26,650 | 34 % | 3.7 % | 0.035 | 0.605 | −0.03 | 0.81 | 0.41 | 0.75 | 9 % |
| | 143 – 152 | 27,713 | 29 % | 3.0 % | 0.035 | 0.570 | −0.07 | 0.75 | 0.44 | 0.75 | 9 % |

## Reading

**1. The global score is the average of two different models.** Product
`W`, "identity absent" and "device missing" are one population: every `W`
row has no identity record and 99 % of no-identity rows are `W`. On it —
82 % of transactions, half of the fraud — PR-AUC is **0.457**; on the other
18 % it is 0.80. The headline 0.637 is not a typical transaction's number;
it is a blend of a segment the model handles well and a segment it does
not. The served policy reaches 89 % of identity-present fraud at block +
review and **67 %** of identity-absent fraud, and blocks 26 % of it against
68 %. This is `docs/error_analysis.md`'s "established card, mainstream
product" failure mode, now with its size: it is the majority of the data.
Every e-mail family below the global number (yahoo, anonymous, apple,
missing) is a family with a high `W` share; nothing suggests the provider
itself carries a separate effect.

**2. Calibration holds across groups.** Mean score tracks prevalence within
about 0.01 in every level with enough rows — product C 0.132 vs 0.126, `W`
0.022 vs 0.020, yahoo 0.023 vs 0.019, debit 0.026 vs 0.023. No group is
scored systematically above its own fraud rate; the sigmoid map fit on
out-of-fold training scores (ADR 0007) does not need per-group correction.
The exceptions are small: discover is *under*-scored (0.092 vs 0.128) and
the > 1000 band over-scored (0.039 vs 0.024, 27 fraud).

**3. The 80 % block bar is a global average, and two levels sit under it
with enough blocks to mean it.** Amount 0 – 25: precision 0.66 on 289
blocks; discover: 0.67 on 57 blocks. Small-amount false blocks are cheap
under ADR 0006's cost model (0.10 · amount + 2), which is why the
cost-optimal threshold tolerates them; the block *bar* is a precision
promise, and here it is not kept for the cheapest transactions. A
per-band block threshold would be a policy change (§4) — a candidate for
the next threshold document, not applied here.

**4. Recall falls with amount.** Block + review reaches 79 – 82 % of fraud
below 50 and 56 % above 1000 (27 cases; block precision 0.50). The
cost-weighted threshold already prices this, but the ranking is weakest
exactly where a miss costs most. This is the one finding that argues for a
feature or a policy change rather than a note: an amount-aware review
budget, or per-entity deviation on high amounts (`docs/error_analysis.md`).

**5. Drift is visible inside the validation month.** PR-AUC 0.718 →
0.605 → 0.570 across the three ten-day blocks, block precision 0.83 →
0.75. Same effect as the rolling backtests (ADR 0008) and the cohort
result in `docs/feedback.md` (0.743 on the first week alone), seen from a
third angle.

## What this changes

- The model card's performance section now states the identity-present /
  identity-absent split next to the global number; the global number alone
  overstates what most transactions get.
- The bar `block_precision ≥ 0.75` in `configs/promotion.yaml` is checked
  globally. Whether it should be checked per amount band is a §4 decision
  logged for the next threshold review.
- Nothing here changes the served model: the segment where it is weak is
  the segment the feature work already targets, and the subgroup table
  is the right place to measure whether that work lands (rerun after any
  promotion; `<small>` and the 30-positive floor keep it honest on the
  reporting window's smaller cells).
