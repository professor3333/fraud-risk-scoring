# ADR 0011 — Scheduled monitoring and alert delivery

**Date:** 2026-09-21 · **Status:** accepted

## Context

The monitoring content is finished (ADR 0009, `docs/monitoring.md`): the
service records every request, every scored row's inputs and every decision;
a reference frozen from the training and validation windows says what normal
was; `scripts/monitor.py` compares a window of live traffic against it and
ends with a list of flags and a status of `ok` / `warn` / `alert`.

Nothing runs it. A human has to remember, on a laptop, to ask. That makes the
signal worth exactly as much as the operator's memory, and it means the
documented loop — traffic → audit trail → monitor → breach → operator →
possibly retrain — is missing its middle. `docs/monitoring.md` has said so
since it was written: "a scheduler and automated alerts remain future MLOps
work."

Two constraints shape the answer:

1. **The audit trail is inside the service.** It is a SQLite file in the
   container (`FRAUD_AUDIT_DB`). On the free Render host it is also
   ephemeral — a restart loses it — and there is no volume to mount and no
   database to point at from outside.
2. **There is one operator and no budget.** A metrics stack (Prometheus,
   Grafana, Alertmanager) is four more services to run and secure for a
   portfolio demo that scores a few hundred rows a day.

## Options

1. **A scheduler that pulls the rows out and computes the report itself.**
   Add an export endpoint (or widen `/audit/recent`), page through the tables
   into CI, run `build_report` there. Keeps all computation on the runner —
   and moves every scored row, with its input snapshot, out of the service to
   a place none of the data contracts cover, to produce numbers the service
   could have produced itself.
2. **Ship the audit trail to a database the scheduler can read** (Postgres,
   Turso). The honest production answer, and the one to take when this stops
   being a free demo. It is also a new managed dependency, a new secret, a
   new failure mode and a cost, before any of the monitoring logic changes.
3. **Ask the service for the report.** One admin endpoint, `GET
   /audit/monitor`, runs the same `fraud.monitor.report.build_report` over
   its own trail and returns it. The scheduler is a cron'd GitHub Action that
   fetches, keeps the report as a run artifact, and decides whether it is
   worth a notification.
4. **Push instead of pull**: the service alerts itself on a timer inside the
   process. No scheduler at all, but the alerting then depends on the thing
   being monitored being alive — the one failure that most needs reporting —
   and a sleeping free instance never fires.

Delivery, separately: e-mail (a secret and an SMTP provider), Slack (a
webhook secret for a workspace this project does not have), or a **GitHub
issue** in the repository the operator already watches.

## Decision

**Option 3, with GitHub issues as the delivery channel.**

- `GET /audit/monitor` (admin: `FRAUD_ADMIN_API_KEY`, disabled outright while
  unset, like `/outcomes` and `/audit/recent`) takes `since` / `until` and
  returns the report. It calls the same `build_report` as the offline script,
  with the reference that sits next to the served artifact, and reads
  outcomes as of the label feed's clock, so eventual metrics stay confined to
  closed cohorts (ADR 0009). No prediction rows leave the service.
- `scripts/monitor.py --from-url <service> --window-hours 24` fetches that
  report instead of opening a local database, and writes the same
  `reports/monitoring/<name>.json` and `.md`.
- `.github/workflows/monitor.yml` runs it daily (07:10 UTC) and on demand,
  uploads the report, and runs `scripts/alert.py`.
- `configs/alerting.yaml` holds the orchestration: the window, a **volume
  floor** (`min_rows: 200`), the severity that notifies (`alert_on: alert`)
  and the issue's title and label. Detection thresholds stay in
  `fraud.monitor.report`, so they cannot differ between an offline run and
  the service.
- Alerting is one issue per **episode**, not per run: a breach opens a
  labelled issue, further breaches comment on it, and the first healthy run
  closes it. The close is the recovery signal.

Two decisions inside that are worth stating on their own, because they are
the difference between an alert channel an operator keeps and one they mute:

- **A window below `min_rows` predictions is `no_data`, never an alert.** The
  public demo is idle for days. A PSI over forty rows measures the sample
  size, not the traffic; reporting it as drift would train the operator to
  ignore the channel. The run is still recorded, with its summary.
- **`warn` does not page.** A single PSI between 0.10 and 0.20 happens on
  ordinary days — `n_missing_transaction` alone does it whenever provider
  coverage shifts (`docs/monitoring.md`). `alert` means a PSI past 0.20, an
  error rate over 5 %, a label-feed gap, or an eventual metric below its
  reference; those are the four things worth an interruption.

## Consequences

- The loop in `docs/monitoring.md` is closed to the point of *notification*.
  It deliberately stops there: nothing here retrains, promotes or rolls back
  a model. A drift flag is not by itself evidence that a new model would be
  better (`docs/retraining.md`), and an automatic promotion would bypass ADR
  0010's gates.
- **What the demo can actually report is bounded by the demo.** Render's free
  instance sleeps and loses its SQLite file on restart, so a window may
  legitimately contain nothing, and a report is over whatever survived since
  the last restart. The scheduled job therefore proves the mechanism, not a
  month of uninterrupted production monitoring; `docs/monitoring.md` says so
  in the same words.
- The job needs `vars.RENDER_URL` and `secrets.FRAUD_ADMIN_API_KEY` matching
  the service's own `FRAUD_ADMIN_API_KEY`. Without either it exits 0 having
  done nothing: a fork, or a demo taken down, is not an incident. Until that
  secret is set on both sides, the workflow is configuration-inert — it runs
  and reports that there is nothing to monitor.
- The monitoring endpoint costs CPU on a 0.1-CPU host, so the window is
  counted before it is loaded (`monitor_max_rows`, default 100,000 rows →
  413 and a request for a shorter window). A day of demo traffic is three
  orders of magnitude below that.
- The API section of the report now **excludes the admin surface**
  (`/audit/*`, `/outcomes`). Those calls are the monitor's own and the label
  feed's; counting them meant a refused monitoring request raised the error
  rate that the next monitoring request alerted on.
- `psi` now sums over sorted keys. Python randomises string hashing per
  process, so the same window computed in the service and in an offline run
  differed in the last floating-point place; a test now asserts the two
  reports are equal, which only holds if the sum is ordered.
- Using GitHub issues buys the notification channel for free (the operator
  already watches the repository; e-mail and mobile notifications are
  GitHub's) at the cost of an alert history that lives in a public
  repository. Report bodies carry aggregates only — counts, shares, PSIs,
  metrics — never a transaction, an identifier or a score.
- Not done: paging (no on-call exists), an SLO or burn-rate alert (the demo
  has no traffic to base one on), alert suppression windows beyond the
  episode rule, and automatic retraining. Delivery to Slack or e-mail is a
  second `deliver` implementation in `scripts/alert.py` when a channel
  exists.

## Files

`src/fraud/monitor/alerts.py` (the decision), `src/fraud/monitor/remote.py`
(the fetch), `GET /audit/monitor` in `src/fraud/serve/app.py`,
`scripts/monitor.py --from-url`, `scripts/alert.py`, `configs/alerting.yaml`,
`.github/workflows/monitor.yml`, `tests/test_alerting.py`, the monitoring
tests in `tests/test_serving.py`, and `docs/monitoring.md` → Scheduled
monitoring.
