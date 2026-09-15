# Monthly retraining lifecycle (simulated offline)

`uv run python scripts/retrain.py` → `reports/retrain/lifecycle.csv`,
`models/retrain/cutoff_<T>/` (artifact, frozen golden, `decision.json`).
MLflow experiment `fraud-retrain`. Not scheduled; this reproduces the
lifecycle offline so every step exists and is tested before any automation.

```
new month arrives (labels through day T are mature)
  ↓ training = days ≤ T − 30           (all mature months but the latest)
  ↓ validation month = days T − 29 … T (the latest mature month)
  ↓ challenger: fit preprocessing + XGBoost on training; calibrate on OOF
    folds strictly inside training (sigmoid, ADR 0007)
  ↓ incumbent (last promoted artifact) scored on the same validation month
  ↓ promote if challenger PR-AUC ≥ incumbent + 0.005, else keep the incumbent
  ↓ re-select the block threshold at the 80 % precision bar on the month
  ↓ freeze the serving artifact with a 50-row golden (fraud.serve.parity)
```

No step reads days > T. Cut-offs at 120 and 150 use development data only;
`--include-reporting-window` adds a cut-off at 183 and must be logged as a
test-window consultation (ADR 0002).

## Result (challenger recipe E022)

| cut-off | validation month | positives | incumbent | challenger | decision | block threshold | serving PR-AUC |
|---:|---|---:|---:|---:|---|---:|---:|
| 120 | days 91–120 | 3,898 | — | 0.6146 | promote (bootstrap) | 0.47 | 0.6146 |
| 150 | days 121–150 | 2,850 | **0.5216** | **0.6467** | promote (+0.125) | 0.425 | 0.6467 |

Reading:

- **Staleness is expensive.** The incumbent trained through day 90 scores
  0.522 on days 121–150; the challenger trained through day 120 scores
  0.647 on the same month. One month of extra data — and one month less
  distance — is worth **+0.125 PR-AUC**, more than every feature and tuning
  experiment in this project combined (E003 → E022 is +0.067). This is the
  same effect the rolling backtests measure as the gap-0 vs gap-30
  difference (`docs/backtest.md`).
- **The block threshold moves** (0.470 → 0.425) because the score scale
  shifts between months; freezing it would silently change the block
  precision, which is what the reporting window showed happening (0.80 →
  0.71). Re-selecting it is part of the cycle.
- **Promotion is not automatic**: the +0.005 margin is the guard against
  promoting noise; on this data every cycle clears it by two orders of
  magnitude, which says monthly is, if anything, not frequent enough.

## What would make this a scheduled job

1. A label feed: the cycle assumes labels through day T are mature. In
   production that is "reports received by T + reporting window"; the
   cut-off must trail the calendar by that window.
2. A trigger: the monitor's labelled report (`docs/monitoring.md`) drops
   below the reference → run the cycle; or simply monthly.
3. Deployment: the promoted artifact + golden replace the served files;
   the service refuses to start on a parity mismatch, so a bad promotion
   cannot serve. `scripts/deploy_check.py` is the post-deploy gate.
4. Registry: the cycle's `decision.json` files are the promotion log; an
   MLflow model registry stage transition would be the production form.
