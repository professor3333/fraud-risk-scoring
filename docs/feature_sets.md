# Feature sets: cumulative ladder and family cuts (E018 – E020)

Evidence: `reports/feature_sets/ladder.csv`, `reports/feature_sets/family_cuts.csv`
(`scripts/feature_ladder.py`, MLflow `fraud-feature-sets`). All fits use the
E008 parameters; all numbers are validation PR-AUC on the frozen split;
seed noise is 0.002 sd (ADR 0003).

## 1. Cumulative ladder

| set | inputs | val PR-AUC | Δ step | recall @ P ≥ 0.90 | gap |
|---|---:|---:|---:|---:|---:|
| F0 raw | 423 | 0.6014 | — | 0.307 | 0.30 |
| F0 + F1 missing counts | 426 | 0.6043 | +0.003 | 0.300 | 0.29 |
| + F2 amount structure | 431 | 0.6041 | −0.000 | 0.305 | 0.30 |
| + F3 time of day | 433 | 0.6003 | −0.004 | 0.303 | 0.31 |
| + F4 e-mail flags / families | 438 | 0.6033 | +0.003 | 0.310 | 0.31 |
| + F5 card × address / e-mail keys (frequency) | 442 | **0.6166** | **+0.013** | 0.341 | 0.31 |
| + F6 frequency tables | 454 | **0.6255** | **+0.009** | 0.321 | 0.31 |
| + F7 card+address history (earlier rows) | 462 | 0.6191 | −0.006 | 0.333 | 0.32 |

Reading: F1 – F4 wander within noise (±0.004); F5 and F6 are the only
steps larger than the noise floor, and they are the same two sets the
one-at-a-time experiments accepted (E016, E006). F7 costs a little, as in
E007/E017. The ladder and the per-set experiments agree.

**"+F6" at 0.6255 looked like a new best** (shipped E016 = F0+F5+F6 scores
0.6190). Seed-paired (E019): 0.6255 / 0.6156 / 0.6214 vs 0.6190 / 0.6221 /
0.6205 → mean **+0.0003**. It was a lucky seed. Accumulating F1 – F4 on top
of the shipped set neither helps nor hurts measurably — at depth 8 the
extra 15 split candidates are absorbed.

## 2. Family cuts — Vesta's features versus mine

`raw` = the provider's columns as delivered (F0). `shipped` = raw + my
frequency tables + composite keys (E016). `V` = Vesta's 339 engineered
columns; "Vesta-engineered" = `C`, `D`, `M`, `V` together.

| cut | inputs | val PR-AUC | recall @ P ≥ 0.90 |
|---|---:|---:|---:|
| raw, transaction only (no identity, no `V`) | 50 | 0.5864 | — |
| raw, transaction + identity (no `V`) | 84 | 0.5913 | — |
| raw, everything incl. `V` (F0) | 423 | 0.6014 | 0.307 |
| shipped, transaction only (no identity, no `V`) | 62 | 0.6035 | 0.274 |
| shipped except `V` | 100 | **0.6214** | 0.320 |
| shipped, everything | 439 | 0.6190 | 0.329 |
| shipped except Vesta-engineered (`C`, `D`, `M`, `V`) | 62 | **0.4325** | 0.129 |

Three answers to "how much comes from my features versus Vesta's?":

1. **Vesta's `V` block is worth ≈ 0.01 on raw columns and ≈ 0 on the
   shipped set.** Raw: 0.5913 → 0.6014 (+0.010). Shipped: 0.6214 without
   `V` vs 0.6190 with it (seed-paired, E020: −0.0025 — noise). Once the
   frequency tables and composite keys exist, the 339 `V` columns add
   nothing the model can use. A **100-input model at the shipped score** is
   available (`configs/features/f5_noV.yaml`): 4× fewer inputs, faster to
   train and serve. Not shipped, because the rule accepts score gains, not
   size gains; it is the obvious choice if serving cost ever matters.
2. **My features are worth ≈ 0.02 – 0.03 wherever they are added.** Raw
   transaction-only 0.5864 → shipped transaction-only 0.6035 (+0.017);
   raw no-`V` 0.5913 → shipped no-`V` 0.6214 (+0.030); F0 0.6014 → shipped
   0.6190 (+0.018).
3. **Vesta's other engineered families (`C`, `D`, `M`) are the foundation:
   removing all four costs 0.19** (0.6190 → 0.4325). The provider's counts
   and time deltas — which summarise each card's history at authorization
   — are what makes a 0.6 model possible; nothing computable from the
   remaining raw fields plus my frequency tables recovers them (E007 and
   E017 tried the history route and found the provider had already done
   it). Identity adds ≈ 0.005 on raw columns and ≈ 0.02 on the shipped set.

The honest summary for the README: on this dataset the provider's
engineering carries the model; the engineering done here adds ~0.02 PR-AUC
on top, most of it from asking "how common is this card at this address /
with this e-mail in the training population", and the largest single block
of provider columns (`V`) turns out to be optional once that question is
asked.

## 3. What was rejected here

- E019 cumulative F0 – F6 as a candidate: +0.0003 seed-paired → not a change.
- E020 shipped-minus-`V` as a candidate: −0.0025 seed-paired → score-neutral;
  kept as the compact option, not shipped.
