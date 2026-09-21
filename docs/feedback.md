# The delayed-label feedback loop

A fraud label is not known when a transaction is scored. The chargeback
comes in days or weeks later; "legitimate" is only ever the absence of a
report by the time the reporting window closes — 120 days under the host's
own labelling rule (ADR 0001). The system now carries that lifecycle end to
end (ADR 0009):

```
prediction ──► action ──► (time passes) ──► outcome arrives ──► attached to the prediction
                                                                       │
                     serving ─► prediction_events (…, transaction_dt)  │
                     POST /outcomes  or  scripts/feedback.py ─► outcomes (label, event_dt, observed_dt)
                                                                label_feed (the clock each delivery reached)
                                                                       │
                     scripts/monitor.py ──► per transaction: matured | pending | overdue
                                            eventual metrics on CLOSED cohorts only
                                            early signal on open cohorts
                                                                       │
                     scripts/retrain.py --label-maturity-days D ──► trains on the closed cohort;
                                            the served month is scored once its labels mature
```

## Three states, not two

At any moment a scored transaction is one of:

| state | meaning | what it may be used for |
|---|---|---|
| **matured** | its outcome has arrived — a report, or the window closed with none | eventual metrics, if its cohort is closed; the early signal otherwise |
| **pending** | no outcome and the window is still open | nothing; expected |
| **overdue** | no outcome and the window has closed | nothing; the feed has a gap → alert |

A *cohort* (all transactions from one day) is **closed** once
`transaction_dt + 120 days ≤ now`. Only closed cohorts have both classes;
until then every label that has arrived is a positive.

## The simulation

The dataset has the labels, not the dates they became known.
`configs/feedback.yaml` supplies them: a fraud row's report lag is drawn
from a log-normal (median 21 days, σ 0.8; 90th percentile ≈ 58 days), a
legitimate row is confirmed at day 120. `scripts/simulate_feedback.py`
scores the validation month (days 123 – 152, 85,044 transactions) through
the service's own scoring and audit path into
`models/audit/feedback_sim.sqlite`, then advances the feed clock and runs
the monitor at each step. Nothing after day 152 is scored; the clock just
lets outcomes arrive. Reports: `reports/feedback/day_<clock>.md`, summary
`reports/feedback/timeline.csv`.

| clock (day) | days after window | matured | pending | closed cohort | fraud reported so far¹ | positive rate, arrived labels | positive rate, closed cohort | early recall block / block+review | eventual PR-AUC | block precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 152 | 0 | 991 | 84,053 | 0 % | 34 % | **100 %** | — | 0.51 / 0.80 | — | — |
| 160 | 8 | 1,511 | 83,533 | 0 % | 52 % | **100 %** | — | 0.49 / 0.79 | — | — |
| 182 | 30 | 2,358 | 82,686 | 0 % | 82 % | **100 %** | — | 0.48 / 0.78 | — | — |
| 212 | 60 | 2,711 | 82,333 | 0 % | 94 % | **100 %** | — | 0.47 / 0.78 | — | — |
| 242 | 90 | 2,813 | 82,231 | 0 % | 98 % | **100 %** | — | 0.47 / 0.78 | — | — |
| 250 | 98 | 27,102 | 57,942 | 30 % | 99 % | 10.5 % | 3.4 % | 0.42 / 0.76 | **0.743** | 0.86 |
| 260 | 108 | 52,992 | 32,052 | 61 % | 99 % | 5.4 % | 3.5 % | 0.42 / 0.73 | **0.680** | 0.82 |
| 272 | 120 | 85,044 | 0 | 100 % | 100 % | 3.4 % | 3.4 % | — | **0.637** | 0.80 |

¹ Simulation-only (needs the final labels): the share of the month's
eventual fraud whose report has arrived.

Reading:

- **For 90 days the arrived labels are 100 % fraud.** Nothing is wrong with
  the feed; that is what a chargeback stream looks like before the window
  closes. Any positive rate, precision or PR-AUC computed on "labels so
  far" during that time is computed on a sample with no negatives. The
  monitor prints the arrived-label rate next to the closed-cohort rate so
  the gap is seen, and evaluates only the latter.
- **The early signal is usable from day one — and it drifts down as slower
  reports arrive.** Of the fraud reported by the window's last day, 80 %
  had been blocked or reviewed; of all the month's fraud, 78 %. Quickly
  reported fraud is slightly easier fraud. It is a leading indicator of
  recall with a known optimistic tilt, and says nothing about false
  positives.
- **The first eventual number for a month is its best.** Cohorts close
  oldest-first, and the oldest days are the closest to the training
  window: PR-AUC 0.743 on days 123 – 130 alone, 0.680 with 61 % of the
  month closed, 0.637 for the full month (the reference value, as it must
  be — same rows). A monitor that alerts on the partial eventual number
  would first say the model improved.
- **The month has no eventual metric until day 243, and no complete one
  until day 272.** With the host's rule, production monitoring of a
  freshly deployed model shows drift in score and input distributions
  immediately, and in performance four months later. This is why the
  score / feature PSI in `docs/monitoring.md` exists, and why its
  reporting-window demonstration (eventual PR-AUC 0.617 one month out)
  could only ever be produced offline.
- **Overdue is the feed's health check.** The simulation delivers
  everything, so `overdue` stays 0; drop a delivery and the report status
  goes to alert (`test_monitoring`).

Report lag of the arrived fraud is printed too (median 18 d at day 182,
21 d once complete) — in production this is the number to fit the
simulation's log-normal to, and the one that tells you when the early
signal has stopped moving.

## Retraining waits for maturity

`docs/retraining.md`'s lifecycle assumed labels through the cut-off were
final. With a maturity *D* the cycle at calendar day *T* may fit on labels
through *T − D*: training ends at *T − D − 30*, the validation month is the
one before *T − D*, and the artifact then serves *T + 1 … T + 30*. The
lifecycle now records that served month's eventual PR-AUC and block
precision (scored after the fact, informing nothing in the cycle) and the
**staleness** — days from the end of training data to the end of the
served month.

Under the host's 120-day rule the first cut-off with a month to train on
is day 180, trained through day 30; the lifecycle refuses earlier cut-offs
with "nothing to train on". The sensitivity run below uses *D = 30*
(`uv run python scripts/retrain.py --label-maturity-days 30`,
`reports/retrain/delay30/lifecycle.csv`; MLflow `fraud-retrain`).

| maturity *D* | calendar cut-off | labels final through | trained through | validation month | validation PR-AUC | serves | **served-month PR-AUC** | block precision in service¹ | staleness |
|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| 0 (`docs/retraining.md`) | 120 | 120 | 90 | 91 – 120 | 0.615 | 121 – 150 | **0.522** | 0.71² | 60 d |
| 30 | 120 | 90 | 60 | 61 – 90 | 0.655 | 121 – 150 | **0.477** | 0.69 | 90 d |
| 30 | 150 | 120 | 90 | 91 – 120 | 0.615 (incumbent 0.547 → promote) | 151 – 180 | not scored (reporting window) | — | 90 d |

¹ The threshold was chosen at the 80 % precision bar on the validation
month; this is what it delivered on the served month. ² From the D = 0
run's next cycle, where the same artifact is the incumbent scored on the
same month (`reports/retrain/lifecycle.csv`); the D = 0 run predates the
served-month column.

Reading:

- **Waiting for labels costs another 0.045 PR-AUC on the month that
  matters.** The same month (121 – 150) is served by a model trained
  through day 90 when labels are final on the day, and through day 60 when
  they take 30 days: 0.522 vs 0.477. Under the host's 120-day rule the gap
  is a further three months of staleness, which the data is too short to
  measure and there is no reason to expect to be kinder.
- **The number the cycle selected on and the number the month got are
  0.18 apart** (0.655 on days 61 – 90, 0.477 on days 121 – 150). The
  validation month is, by construction, the freshest month with final
  labels — and the served month is the one furthest from it. The
  lifecycle now writes both, so a promotion decision can be read against
  what it delivered once the labels came in.
- **The 80 % precision bar does not survive the gap either**: 0.69 in
  service. Re-selecting the threshold monthly (the existing rule) limits
  the damage; it cannot remove a lag it cannot see.
- **The cycle itself still works.** At cut-off 150 the challenger trained
  through day 90 beats the incumbent trained through day 60 by +0.068 on
  days 91 – 120 and is promoted; the challenger's 0.6146 is identical to
  the D = 0 run's cut-off-120 challenger, as it must be (same fit, same
  seed).
- What would help, in order: shorten the effective lag by retraining on
  the early-arriving positives *plus* confirmed negatives from older
  cohorts (a censored-label formulation — not attempted, and it must not
  be done by simply training on arrived labels, `mature_rows` exists to
  prevent that); features that age well (`docs/backtest.md` measures
  decay per candidate); a shorter maturity where the business can
  justify one.

## Operating it

```bash
uv run python scripts/simulate_feedback.py --fresh                      # the table above (~1 min)
uv run python scripts/feedback.py --as-of-day 200                       # advance the feed on the served db
uv run python scripts/monitor.py --since … --until …                     # outcomes read as of the feed clock
uv run python scripts/monitor.py --labels final.csv                     # a fully matured label file, as before
uv run python scripts/retrain.py --label-maturity-days 30               # the lifecycle under a maturity
```

In production `POST /outcomes` replaces `scripts/feedback.py`: the
chargeback system delivers `{transaction_id, is_fraud, event_dt,
observed_dt}` batches, and delivers the legitimate outcomes when the
window closes. The endpoint writes the label store, so it is an admin
route: it answers only to `FRAUD_ADMIN_API_KEY` and is closed (403) on a
host where that key is unset, such as the public demo. The monitor never infers a negative: a transaction past the
window with no delivered outcome is *overdue*, not legitimate, so a silent
feed cannot masquerade as a clean month.

What is deliberately not modelled (ADR 0009): label revision after the
window, per-product maturity, and the account-level propagation of the
label (a reported account's later transactions become positives at the
account's report time, not their own).

**Revision is the one of those three that the storage layer, not just the
simulation, would have to change.** `AuditLog.record_outcomes` writes with
`INSERT OR IGNORE`: the first label for a transaction wins and a redelivery
is a no-op, which is exactly right for a feed replaying fixed historical
labels, and exactly wrong for a real one. Chargebacks get reversed,
investigations get overturned, friendly fraud gets reclassified — and a
corrected outcome arrives looking identical to a duplicate, so it would be
dropped in silence and every eventual metric computed afterwards would be
computed from the superseded label.

Fixing it is not an integration project: make `outcomes` append-only with a
revision chain, and decide what the monitor reads. That decision is the
interesting half, because "what is true now" and "what did we know when we
retrained" stop being the same query — the first is latest-wins, the second
is as-of, and a retraining cycle's honesty depends on the second. The
endpoint's `source` field already carries which system said so (chargeback
feed, investigation, analyst), so a multi-source production feed needs no
schema change; a revisable one does.
