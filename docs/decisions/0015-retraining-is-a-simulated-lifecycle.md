# ADR 0015 — Retraining is a simulated lifecycle, not a scheduled loop

**Date:** 2026-09-29 · **Status:** accepted · **Amends:** ADR 0012

## Context

ADR 0012 connected the retraining pieces and put them on a monthly schedule.
The pieces work: `docs/retraining.md` records two cycles run by hand that
reproduce the shipped champion to the bit. The *schedule* could never do
anything, for reasons that are properties of the setting, not bugs:

- **There is no future.** The labelled data ends at day 182. No new
  transactions arrive, so there is nothing new to train on.
- **The label feed is not a training source.** `POST /outcomes` records a
  label per transaction ID. The service keeps a compact monitoring snapshot of
  the inputs, not the ~430 columns the model is fitted on, so outcomes posted
  to the live demo cannot become training rows. Training reads the Kaggle
  snapshot's `isFraud`.
- **A runner has no clock.** The trigger reads the label feed's watermark from
  the audit database. A fresh GitHub runner has none, so the default path
  always decided `WAIT (no-labels)`.
- **Maturity closes the window.** Under the host's 120-day label maturity, and
  with the reporting window (day 153+) protected by ADR 0002, no evaluation
  month exists that the current champion (trained through day 122) has not
  seen. Checked for feed-clock days 1–500 against a champion trained through
  day 122: 179 wait for labels, 93 have too little new data, 228 would read
  the reporting window, and none runs.
- **The job could not start anyway.** The repository has no Kaggle secrets,
  so the workflow exited before downloading anything. `retrain.yml` has zero
  runs.

Calling that "scheduled retraining" overstated the system.

## Options

1. **Keep the schedule and the claim.** Rejected: the claim is false.
2. **Build the live loop.** This needs versioned feature *and* outcome
   snapshots, a durable feed watermark, mature-cohort completeness checks,
   persistent registry storage, and a rolling evaluation policy that does not
   spend the reporting window. It also needs a data source with a future,
   which this project does not have.
3. **Declare the scope honestly.** Make the lifecycle a repeatable
   simulation with an explicit snapshot and clock, run by one command, and
   keep the real single-cycle path for manual use.

## Decision

Option 3.

- `scripts/simulate_lifecycle.py` runs the whole lifecycle from an empty
  workspace, on a declared snapshot (the labelled training data) and a
  declared clock (days 92, 110, 122, 183). It calls the same entry points as
  the workflow: `retrain_cycle.py` (trigger → fit → gates → promote → policy
  patch) and `publish_champion.py`, which publishes to a local release store
  laid out like the GitHub releases. After each promotion it starts the
  service the way the host does, from that release URL and the patched config,
  so the digest pin, the band pairing and the parity check all run. It shows
  both valid skips: too little new data (day 110) and the reporting window
  (day 183).
- **One stated assumption.** Labels are final on the day they occur
  (`--label-maturity-days 0`). The Kaggle labels already are; under the
  120-day rule the dataset yields no cycle at all.
- **The simulation is isolated.** It never touches the served champion, the
  real registry, `configs/serving.yaml` or GitHub. MLflow artifacts are
  created inside the workspace (MLflow otherwise writes them into the
  repository's `./mlruns`, which the first run did).
- `retrain.yml` loses its cron. Manual dispatch defaults to `mode: simulate`.
  `mode: cycle` keeps the real single cycle, and requires an explicit
  `as_of_day`.
- A fixture test runs the simulation end to end in CI in about 20 s.

## Evidence

A fresh clone with no `models/`, no MLflow store and no data cache, on
2026-09-29, `uv run python scripts/simulate_lifecycle.py`, production recipe,
17.6 minutes, exit 0 (`reports/retrain/simulation.md`):

| day | trigger | outcome | published | service started from it |
|---|---|---|---|---|
| 92 | bootstrap | promoted | `champion-611199307f76` | yes, bands 0.0853 / 0.51 |
| 110 | too-little-new-data (18 of 30 days) | waited | — | — |
| 122 | calendar | promoted, PR-AUC +0.0754 over the incumbent on days 93–122 | `champion-ece2ed287853` | yes, bands 0.0837 / 0.48 |
| 183 | reporting-window | waited | — | — |

Each promotion produced a reviewable `configs/serving.yaml` diff (bands plus
`champion_sha256`). The day-122 champion's re-derived bands, 0.0837 / 0.48,
are the ones the 2026-09-21 hand run recorded for the same cut-off
(`docs/retraining.md`), so the simulation drives the same path. The clone's
working tree gained one file, the report.

## Consequences

- The README and PROGRESS no longer say retraining is scheduled. They say
  what exists: an automated lifecycle, demonstrated on a simulated clock and
  runnable by hand.
- The live-loop requirements listed under option 2 are the work a real
  service would need. None of them is started here.
- Monitoring stays scheduled (ADR 0011): it has a real source, the live
  service's own audit trail.
