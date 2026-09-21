# ADR 0004 — Entity reconstruction and history features

**Date:** 2026-09-14 · **Status:** accepted as the legitimate way to build
these features — then **built, measured and rejected at the capacity of the
day** (E007, E017), and in **E025 (2026-09-21) found to pay at the shipped
model's capacity**, on new cards rather than established ones. They are still
**not in the served model**, which is row-local, and nothing is promoted. Read
*Outcome* below before reading the rest as a description of what this system
does.

## Context

`D1` behaves as "days since this card was first seen" (`docs/eda.md` §6):
`day − D1` is constant over a card's life, and `(card1, day − D1)` yields
≈150k candidate card entities. 81 % of fraud rows sit in entities with ≥ 2
fraud rows because the label propagates within an account (ADR 0001).

This is the dataset's biggest lever and its biggest trap. Kaggle solutions
build a "uid" this way and then aggregate *everything* over it — including
rows from the future and, implicitly, the label. Under G11 each piece must be
argued separately: what it exploits, whether it exists at authorization time,
and whether it survives G2 / G3.

## Options

1. **No entity features.** Safe, leaves the largest known signal on the table.
2. **Entity key as a feature** (`day − D1`, or the key's frequency over the
   whole data). Rejected: `day − D1` is an absolute date, so it encodes
   *when* a card was first seen — a card born on day 150 exists only in
   validation/test. Whole-data counts are a G2 violation.
3. **Per-entity aggregates over all rows** (mean amount over the card's
   whole history including later rows). Rejected: uses the future.
4. **Per-entity aggregates over strictly earlier rows only**, no label
   involved: how many transactions this card has made before now, how long
   since its last one, how its current amount compares with its past. What
   they exploit: velocity and deviation from the card's own habit — the
   classic fraud signals a real system computes from its transaction log.
   Available at authorization: yes, given a history store. Survives G2 (no
   fitting) and G3 (earlier rows only, enforced by construction and by test).
5. **Prior-fraud-on-this-entity features** (count of earlier rows labelled
   fraud). Rejected outright: the label arrives weeks later in production, and
   because of propagation this feature *is* the label for most positives. It
   would produce a spectacular validation score that means nothing.

## Decision

**Option 4.** Entity key `(card1, addr1, day − D1)`, used **only as a grouping
key, never as a feature**. Rows with any key component missing get no
entity features (NaN). Features, each computed from rows of the same entity
with a strictly earlier `TransactionDT` (ties broken by `TransactionID`):

| feature | definition |
|---|---|
| `ent_prior_count` | number of earlier transactions of the entity (0 for the first) |
| `ent_seconds_since_prev` | `TransactionDT` minus the previous transaction's; NaN for the first |
| `ent_prior_amt_mean` | mean `TransactionAmt` of earlier transactions; NaN for the first |
| `ent_amt_ratio` | `TransactionAmt / ent_prior_amt_mean`; NaN for the first |
| `ent_prior_count_1d` | earlier transactions within the previous 86 400 s |

No label enters any of them. `addr1` is part of the key so that a card
number reused across billing regions is treated as separate accounts
(`card1` alone has a median of 2 distinct start-days per value).

**Where it runs.** These features need the entity's past rows, so they are
computed by `fraud.features.history.add_entity_history` over the full
time-ordered frame *before* the split, as a data-preparation step — the
pipeline then treats them as input columns. This is a deliberate exception
to "everything inside the sklearn pipeline": the function is stateless and
fits nothing, so G2 is intact, and the G3 guarantee is tested on a synthetic
frame in which later rows are altered and earlier rows' features must not
change.

**Serving contract (binding for Stage 6).** The API must obtain these five
columns the same way: from the entity's transaction log up to the request
time, using the same function. The demo service keeps an in-memory history
table built from the labelled data and computes the features for a request
incrementally; a parity test asserts the incremental value equals the batch
value for held-out rows. If that parity cannot be achieved, the features are
dropped from the served model rather than approximated.

## Consequences

- E007 measures the gain. Expected to be the largest single step in the
  project; if it is not, the reconstruction hypothesis is wrong.
- `docs/leakage_audit.md` gains a row per feature; the key is documented as
  **out** as a feature.
- Serving becomes stateful (a history store). This is what any real fraud
  system has; the model card states it plainly.
- Train-window rows early in the data have short histories, validation and
  test rows have long ones. That is the true production situation (a deployed
  model always sees more history than it was trained on) and is *not* a
  leak, but it is a shift worth watching in the ablation.

## Outcome (added 2026-09-21)

The decision above was about *legitimacy*: option 4 is the only construction
of entity history that survives G2 and G3, and that judgement still stands.
It says nothing about whether the features are worth having, and the
consequences section's "expected to be the largest single step in the
project" turned out to be wrong. Recorded here because this file is the one
a reader reaches first, and until now it read as a description of the
shipped system.

**What was measured.** Two entity definitions, eight features, four rounds
of experiments:

| experiment | entity | features | validation Δ | verdict |
|---|---|---|---:|---|
| E007 | `(card1, addr1, day − D1)` | the five above | −0.002 seed-paired | reject |
| E017 | `card1 + addr1` (no `D1`) | + expanding std / max | −0.001 | reject |
| F7 on the ladder | as E007 | five, on top of F0–F6 | −0.006 (462 cols) | not shipped |
| E025 | `card1 + addr1` | eight, at E022 capacity | **+0.0066** aggregate; **+0.043** on `W` ∩ `D1 < 14`; −0.012 on `W` ∩ `D1 >= 14` | over the rule — not promoted, see below |

**Why.** The provider's `C*` columns are counts of entities linked to the
card and `D*` are time deltas — the provider had already summarised each
card's past at authorization time, presumably by this same route. Once the
history is restricted to what a live system can compute (label-free
aggregates over strictly earlier rows), there is nothing left to add **for a
card the provider has had time to summarise** — E025 below is the qualifier
that sentence needed.
`docs/ablation.md` §2 reaches the same conclusion from the other direction:
`C` and `card` rank high on both permutation and ablation, and they are
irreplaceable. The Kaggle gains attributed to "uid" features came from
aggregates over the whole dataset — future rows, and through label
propagation the label itself — evaluated on a test set sharing those
entities. None of that exists at authorization. This is the project's most
important negative result.

**What this means for the serving contract.** It is not in force, because
the features are not served. Every model input is in the request, the
score of a transaction depends on nothing but its own row, and the startup
parity check covers exactly that (`docs/model_card.md`). The contract above
stays written down as the bar any future history feature has to clear
first — a feature store plus incremental-vs-batch parity, or the feature is
dropped rather than approximated.

**What stays in the repo.** `src/fraud/features/history.py`,
`configs/features/v3_history.yaml`, and the G3 synthetic-frame test that
proves a later row cannot change an earlier row's features. They are the
reference implementation for a legitimate variant, not dead code awaiting
deletion.

**The question this left open, and what answering it found (E025,
2026-09-21).** The aggregate was flat; the slice, it turned out, was not —
but neither was it the slice this file expected. E025 re-ran the features at
E022's capacity on product `W` with `D1 >= 14`, and:

- **The slice chosen was the strong one.** `W` ∩ `D1 >= 14` scores 0.798,
  as well as the identity-present segment. Product `W`'s 0.457
  (`docs/subgroups.md`) lives entirely in cards first seen under a fortnight
  ago, which score **0.211**. `docs/error_analysis.md` pointed at
  established-account takeover because it reasoned from *high-confidence*
  false negatives, and those are established by construction: a stolen card
  with a long clean history produces a confident wrong answer, a new card
  produces an uncertain one that never enters that set.
- **The features help on the opposite population.** +0.043 seed-paired on
  `W` ∩ `D1 < 14`, every seed — the largest feature effect in this project —
  and −0.012 on the established slice. Velocity on a card with almost no
  history is what `C*` / `D*` cannot summarise for a card they have barely
  seen; for an established card they already do.
- **E007 and E017 are bounded, not overturned.** Both measured at E016's
  capacity (depth 8, 800 trees). At E022's (depth 12, 1,600 trees) the same
  family is **+0.0066** on the whole validation window — over the +0.005
  rule. The null was true where it was measured. Nothing re-ran the feature
  question after capacity changed, which is how this sat between two
  experiments for four months.

So the "rejected on evidence" verdict above is correct *for the shipped
recipe at the time it was measured*, and is now bounded rather than general.
Nothing is promoted: the aggregate gain is just over the rule against a seed
sd of 0.002, ADR 0010's gates and the ADR 0008 backtests have not run, the
test window is untouched, and the **serving contract in this file becomes
binding the moment these features enter a served model** — a history store
and incremental-vs-batch parity, or they are dropped rather than
approximated. Full result and caveats: `docs/experiments.md` → E025.
