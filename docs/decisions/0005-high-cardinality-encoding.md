# ADR 0005 — Encoding high-cardinality categoricals

**Date:** 2026-09-14 · **Status:** accepted

## Context

Twelve columns are identifiers or free-text-like with 59 – 13,553 levels:
`card1` (13,553), `card2` (500), `card3` (114), `card5` (119), `addr1`
(332), `addr2` (74), `P_emaildomain` (59), `R_emaildomain` (60), `DeviceInfo`
(1,786), `id_30` (75), `id_31` (130), `id_33` (260). The numeric ones enter
the current best (E003) as raw numbers — an arbitrary ordering that trees can
still split on, but that carries no notion of "how common is this value". The
string ones are excluded entirely (leakage audit).

Kaggle solutions lean heavily on count / frequency features for exactly these
columns. Under G11 the question is what they exploit and whether it is
available at prediction time.

## Options

1. **Frequency encoding fit on the training window.** Replace (or add beside)
   each value its share of training-window rows. Unseen values at transform
   time map to 0. What it exploits: rare cards / devices / domains behave
   differently from common ones. Available at prediction time: yes — the
   lookup table is a fitted artefact, frozen at training. Fit scope: training
   window only, inside the pipeline. Survives G2 and G3.
2. **Frequency encoding over the whole dataset** (the Kaggle version).
   Rejected: counts include validation and test rows — G2 violation, and in
   production the count of a card *in the future* does not exist.
3. **Target encoding.** Encodes each level's training fraud rate. Powerful and
   leak-prone: must be computed out-of-fold *and* time-ordered inside the
   training window or it memorises the label. Deferred; only revisited if
   option 1 leaves a measurable gap and the extra machinery can be tested.
4. **Native categorical splits in XGBoost** (`enable_categorical`). Vocabulary
   fit on train; 13.5k-level `card1` becomes a partition search per split,
   which overfits rare levels and is slow. Deferred.
5. **Grouped one-hot with a minimum frequency.** Fine for ≤ 100 levels; blows
   up for `card1` / `DeviceInfo`. Not pursued.

## Decision

**Option 1.** A `FrequencyEncoder` transformer, fit on the training window
inside the sklearn pipeline, adds one `freq__<col>` column per configured
column. Missing values are a level (their own training-window share), because
missingness is informative here. Unseen levels map to 0. The raw numeric
columns stay as they are, so the experiment measures the *added* value of
frequency alone.

Experiment E006 applies it to all twelve columns at once — one change, one
mechanism. If accepted, per-column ablation happens in Stage 5.

## Consequences

- `configs/features/v2_freq.yaml` lists the columns under `frequency:`.
- `tests/test_pipeline.py` asserts the table is learned from training rows
  only, that unseen levels map to 0, and that NaN gets its own share.
- Serving needs nothing extra: the table lives inside the fitted pipeline.
- The table describes the training window's population. Drift in which cards
  or devices are common is a monitoring concern, recorded in the model card.
