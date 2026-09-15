# Monitoring

The offline analysis showed drift (validation PR-AUC 0.637 → reporting
window 0.561; calibration and block precision degrade with time). The
service now carries what is needed to see it happen: an audit trail with
requests and input snapshots, a frozen reference of "normal", and a report
that compares a window of live predictions against it.

```
serving ─► prediction_events (score, action, policy, model, latency)
        ─► input_features    (product, card type, identity, missingness, amount, hour)
        ─► requests          (every call incl. errors, latency, rows)
                     │
                     ▼
scripts/monitor.py  ──  reference (frozen from train + validation)  ──►  report .md/.json
                          + optional labels  →  eventual PR-AUC / precision / calibration
```

## What is monitored

| area | signals | source |
|---|---|---|
| API | requests, requests/s, rows scored, latency p50 / p95 / max, errors and error rate, per endpoint | `requests` table (a middleware records every `/predict*` call, successful or not) |
| Predictions | mean / p50 / p99 probability, score PSI against the validation distribution, block / review / approve shares vs reference, model versions and policies seen | `prediction_events` |
| Data | PSI per monitored input: product, card4, card6, device type, hour, identity presence, address missing, e-mail presence, amount, missing-field counts | `input_features` (a compact snapshot per scored row) |
| Model | score drift, feature drift; with labels: eventual PR-AUC, ROC-AUC, block precision, recall (block, block + review), Brier, ECE, each against the reference | events × labels |

PSI thresholds: ≥ 0.10 warn, ≥ 0.20 alert. Eventual PR-AUC more than 0.03
below reference, block precision more than 0.05 below, or error rate > 5 %
raise an alert. Categoricals are compared on the reference's own level set;
count-like features use equal-width bins (quantile bins collapse on
discrete values).

## The reference

`scripts/monitor_reference.py --run-name xgb_f5_capacity` writes
`models/<artifact>_monitor_reference.json`: feature distributions from the
**training window** (the population the model and its frequency tables were
fit on), and score distribution, action shares and performance from the
**validation window** under the rank policy at the served budget. It is
keyed to the artifact's sha256 and re-frozen with each retrain.

## Demonstration: one validation day, then one reporting-window day

Both days were pushed through the running service as ten CSV batches plus
two malformed requests, then `scripts/monitor.py --since … --until …
--labels …` was run on each window (`reports/monitoring/demo_*.md`). The
second day is a reporting-window consultation (ADR 0002's log, #8).

| | validation day 130 | reporting-window day 175 |
|---|---|---|
| rows / requests / errors | 2,807 / 12 / 2 (16.7 %) | 2,683 / 12 / 2 (16.7 %) |
| latency p50 / p95 (280-row CSV batches) | 3.5 s / 4.8 s | 3.7 s / 4.6 s |
| mean probability (reference 0.036) | 0.030 | 0.040 |
| score PSI · action PSI | 0.021 · 0.001 → ok | 0.014 · 0.000 → ok |
| actions block / review / approve | 1.7 / 7.1 / 91.2 % | 2.1 / 7.5 / 90.4 % |
| feature drift flagged | `n_missing_transaction` (PSI 1.09) | `n_missing_transaction` (PSI 0.29) |
| **eventual PR-AUC** (reference 0.637) | **0.655** (+0.018) | **0.617** (−0.020) |
| block precision (reference 0.80) | 0.89 | 0.81 |
| recall block + review | 0.78 | 0.70 |
| Brier · ECE (reference 0.0185 · 0.0037) | 0.0178 · 0.0110 | 0.0228 · 0.0056 |

Reading:

- **The model-level signals move the way the offline analysis said they
  would.** One month out, eventual PR-AUC is 0.02 below reference, Brier
  is worse, recall at the budget drops from 0.78 to 0.70. Without labels
  the monitor cannot see this — score and action PSI are "ok" on both days
  — which is the point of joining labels when they mature.
- **The data-level flag is genuine.** `n_missing_transaction` takes a few
  discrete values (the `V`-block null-pattern combinations, `docs/eda.md`
  §4); the mix of those patterns differs sharply between days (44 % of day
  130's rows have ≥ 207 missing fields, 1 % of day 175's, 7 % of training).
  Provider feature *coverage* shifts daily; a production alert threshold
  for that feature should be set from its day-to-day variance, not from
  the generic 0.20.
- **The error-rate flag works** (the two malformed requests are 16.7 % of
  twelve calls).
- **Latency is the next engineering item**: 3–5 s for a 280-row CSV batch
  is the per-row `details` construction and per-row snapshot loops, not
  the model; single predictions take ~10 ms.

## Operating it

```bash
uv run python scripts/monitor_reference.py --run-name xgb_f5_capacity     # once per artifact
uv run python scripts/monitor.py --since 2026-09-15T00:00 --until 2026-09-16T00:00
uv run python scripts/monitor.py --since … --until … --labels matured_labels.csv
```

Reports land in `reports/monitoring/<stamp>.md` and `.json`. A scheduler
that runs the labelled report when a month's labels mature, and the
unlabelled one daily, is the natural next step; the retraining lifecycle
(`scripts/retrain.py`) is what an alert should trigger.
