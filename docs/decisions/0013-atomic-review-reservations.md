# ADR 0013 — Review reservations are atomic and never cancelled by rescoring

**Date:** 2026-09-29 · **Status:** accepted · **Amends:** the budget accounting in
`docs/review_policy.md` (per-day rank policy)

## Context

The rank policy gives each transaction day a review budget, shared across every
request that scores that day. Serving implemented it as three separate steps:
read the day's standing reviews from the audit trail, decide this request's
actions from what was left, record the decisions. Two defects followed.

**A race.** SQLite's lock covered each statement, not the sequence. FastAPI
runs synchronous handlers in a thread pool, so one uvicorn process still
serves requests concurrently. With a budget of 1, two concurrent single-row
batches of the same day both read zero standing reviews and both returned
`review`. The "one worker" rule in `docs/deployment.md` did not prevent this.

**Silent cancellation.** Standing reviews were the transactions whose
*latest* audited action was `review`. A single `/predict`, documented as
scoring-only, records a payment-path action (`approve` or `block`). Calling
it on a reviewed transaction therefore freed that transaction's slot, and the
next batch handed the slot to someone else. A later batch containing the
reviewed transaction plus a higher-scored newcomer re-ranked them, and could
send the original case to `approve` without anyone deciding to drop it.

## Options

1. **Keep the read-then-write, run one thread.** Removes the race only while
   nothing else in the process is concurrent. Fragile, and it leaves the
   cancellation problem untouched.
2. **Serialise the decide-and-record section in-process** (a Python lock). Fixes
   the race for one process only.
3. **Make the section a database transaction** (`BEGIN IMMEDIATE` on the audit
   database, inside the existing audit lock). Serialises threads *and* processes
   sharing the file, rolls back on failure, and costs nothing extra.
   Horizontal scaling still needs shared state, but the pattern carries over
   (row-level reservations in a shared database).

On cancellation:

- **a.** Keep "latest decision wins".
- **b.** The budget counts transactions **ever** sent to review that day. A
  standing review stays `review` when rescored (unless it now clears the block
  threshold), and it consumes capacity before fresh rows are ranked.

## Decision

**Option 3 with (b).**

- `AuditLog.reserving()` opens `BEGIN IMMEDIATE` under the audit lock. The
  batch and CSV endpoints compute probabilities *outside* it (inference is the
  slow part), then read capacity, assign actions and record events inside it.
  An exception rolls back every event of that request.
- The day's budget is **work reserved**, not outstanding cases. The service
  has no case-closing workflow, so "outstanding" is not something it can
  know. A review slot, once issued, is spent for the day.
- **A review is never cancelled by scoring.** A single `/predict` never
  writes, reads or frees review state. A batch or CSV that re-sends a reviewed
  transaction keeps it in review (block still takes precedence, since it is a
  pure threshold). A retry of the same request therefore returns the same
  decisions and spends nothing more.

## Consequences

- With a budget of N, no mix of concurrent, retried, single, batch or CSV
  calls produces more than N reviewed transactions per day on one audit
  database. Tests reproduce the race (both requests held inside the window),
  cancellation through `/predict`, cancellation through re-ranking, retry
  idempotence and rollback.
- Re-uploading a file with a *smaller* budget can no longer take reviews back.
  The dashboard's "change the budget, re-apply" flow can only add reviews
  within a day. That is the price of not cancelling cases, and it is the right
  direction for the error.
- Requests that touch the audit database queue behind a reservation. The
  section holds no inference, and on the demo's volumes it lasts milliseconds.
- Still per database: replicas with separate SQLite files keep separate
  budgets, and the free host's ephemeral disk forgets them on restart
  (`docs/deployment.md`). A shared deployment would move the same
  reserve-then-record transaction into its shared store.
- Idempotency is by transaction, not by request: two *different* requests for
  the same transaction are one review, which is what the budget means. There is
  no request-level idempotency key, and none is needed for this guarantee.
