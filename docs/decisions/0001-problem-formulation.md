# ADR 0001 — Problem formulation

**Date:** 2026-09-14 · **Status:** accepted

## Context

The label `isFraud` comes from the competition host (Vesta). Per the host's
description, a transaction is labelled 1 if it was reported as a chargeback
(fraud) and, once an account is reported, its later transactions are labelled
1 as well, whether or not each of them was individually fraudulent. EDA
confirms the consequence: 81 % of fraud rows sit in entities with ≥ 2 fraud
rows and 3,118 entities are entirely fraud (`docs/eda.md` §1, §6).

Kaggle's test set has no labels, so every evaluation window is carved from the
labelled data by time.

## Options

1. **Predict `isFraud` as given** — "this transaction belongs to an account that
   is, or will be, reported as fraudulent." Matches the data; the label is
   partly account-level.
2. **Predict only the first fraud transaction per entity** — closer to "detect
   the fraudulent act", but requires a reconstructed entity id (itself a
   leakage question), throws away ~80 % of positives, and the reconstruction is
   noisy.
3. **Predict at the account level** — aggregate to entities and score
   accounts. Changes the deliverable from a transaction API to an account
   monitor; not the stated system.

## Decision

**Option 1.** The system predicts, for a single online transaction at
authorization time, the probability that `isFraud = 1` as defined by the data
provider.

- **Unit of prediction:** one transaction row (transaction fields plus the
  identity record if the provider supplies one; `has_identity` is part of the
  input contract).
- **Moment:** authorization — before the transaction is approved. Only
  information that exists at that moment may be used: the row's own fields and
  facts derived from *strictly earlier* transactions.
- **Meaning of a positive:** the transaction will be, or already belongs to an
  account that will be, reported as fraud. Operationally this is the right
  target for a decline/review decision: a transaction on a compromised account
  should be stopped even if it is not the specific stolen purchase.
- **Consumer:** an automated decision (approve / send to review / decline) at a
  threshold chosen from a cost model (ADR 0006), so the score must rank well
  across the whole population and be reasonably calibrated in the region where
  the threshold sits.

## Consequences

- Entity-level history is a legitimate and powerful source of signal, *but only
  from earlier rows*. Any per-entity aggregate that sees the current or later
  rows would leak the propagated label. This is why time-aware feature code
  and a written leakage audit are required before any such feature exists.
- Random splits are invalid: rows from the same entity would straddle
  train/validation and the propagated label would be memorised. Splits are
  strictly temporal (ADR 0002).
- The label is not "did this specific purchase use a stolen card", so error
  analysis must not treat a positive on a compromised account's routine
  purchase as a false positive of the model.
- Label maturity: the provider assigned labels with a reporting window after
  each transaction. We assume the labels in the training file are mature for
  all 182 days. If late-window fraud rates look implausibly low relative to
  earlier weeks, this assumption is revisited; the EDA weekly rates (2.9 – 5.1 %
  from week 5 on, with weeks 24 – 26 at 3.7 – 4.3 %) do not currently suggest
  under-reporting at the end.
