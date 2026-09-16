# ADR 0009 — Delayed labels: the feedback loop from outcome to retraining

**Date:** 2026-09-16 · **Status:** accepted

## Context

The system ended at *prediction → action*. The model card notes that
production labels for recent weeks would not be mature, and ADR 0001
records the host's rule that makes this concrete: a transaction is fraud
if a chargeback was reported, and legitimate only if none was reported
**within 120 days**. So in production a scored transaction has no label
when it is scored, a fraud label some days later (when the report comes
in), and a legitimate label only when the 120-day window closes. The
monitor (`docs/monitoring.md`) accepted a fully matured label file, and
the retraining lifecycle (`docs/retraining.md`) assumed labels through the
cut-off day were final. Neither said what happens in the months in
between, which is where a real fraud system spends its life.

The dataset has the final labels but not the dates they became known.

## Options

1. **Leave it as a documented limitation.** Nothing to build; the offline
   numbers stay "slightly optimistic in a time-invariant way".
2. **Attach labels as they arrive and compute metrics on whatever is
   labelled.** Simple and wrong: positives arrive first, so "labels so far"
   is a fraud-enriched sample — for a young cohort it is *only* fraud.
3. **Model label maturity explicitly.** Store outcomes with the time they
   became known; classify each scored transaction as matured / pending /
   overdue; compute eventual performance only on cohorts whose window has
   closed; report the open cohorts as an early signal; make retraining
   wait for maturity. Simulate the arrival dates to exercise all of it on
   the chronological data.
4. **Model label revision too** (a late report flipping a confirmed
   negative). Realistic, but the dataset cannot inform it and it doubles
   the state machine.

## Decision

**Option 3**, with the arrival dates simulated (option 4 deferred).

- **Store.** `outcomes` (transaction, label, event time, observed time,
  source) and `label_feed` (the clock each delivery advanced to) in the
  audit database; every prediction event now carries the transaction's
  own clock so it can be aged. `POST /outcomes` is the production ingress;
  `scripts/feedback.py` is the simulated feed. First label wins; a
  redelivery is a no-op.
- **Maturity** = the host's 120 days (`configs/feedback.yaml`). A missing
  outcome inside the window is *pending*; outside it, *overdue* — a feed
  gap, flagged as an alert.
- **Eventual performance is computed on closed cohorts only.** The
  positive rate of the arrived labels is shown next to the closed-cohort
  rate precisely so the bias is visible, never used.
- **Early signal.** On open cohorts: of the fraud reported so far, the
  share blocked and the share blocked-or-reviewed. Biased towards quickly
  reported fraud and silent on false positives; a leading indicator, not
  a metric.
- **Retraining waits.** At calendar cut-off *T* the lifecycle may use
  labels through *T − maturity*; its windows shift back accordingly and
  the month an artifact then serves is scored after the fact as the
  eventual number. `label_maturity_days: 0` reproduces the earlier
  simulation.
- **Simulated report lag** for positives: log-normal, median 21 days,
  σ 0.8, clipped to the window; negatives are confirmed at 120 days. This
  is an assumption about chargeback timing (card-scheme dispute windows
  run to 120 days; most disputes are filed within one or two statement
  cycles), not a measurement. It is a config value with a seed.

## Consequences

- `fraud.monitor.feedback`, `configs/feedback.yaml`, `scripts/feedback.py`,
  `scripts/simulate_feedback.py`, `POST /outcomes`; `docs/feedback.md` has
  the simulation and its reading.
- Under the host's 120-day rule, a month of predictions has **no eventual
  metric for 120 days after its last day**, and its first eventual numbers
  come from its oldest days — the ones closest to training — so they are
  optimistic until the whole month has closed (`docs/feedback.md`).
- Under the same rule the monthly lifecycle cannot complete a cycle inside
  the 183 labelled days (the first cut-off with a month to train on is day
  180, trained through day 30). The retraining simulation therefore uses a
  30-day maturity as a sensitivity, and reports the staleness it implies.
- Positives available before negatives means the store is *usable* early
  only for the early signal; anything that fits — retraining, threshold
  re-selection, calibration — reads the closed cohort (`mature_rows`).
- Deferred: label revision; a maturity that differs by product or card
  type; the propagated (account-level) label arriving with the account's
  first report rather than per transaction.
