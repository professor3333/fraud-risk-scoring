# ADR 0008 — Drift-aware validation: rolling temporal backtests with horizons

**Date:** 2026-09-15 · **Status:** accepted

## Context

ADR 0002's validation window (days 123 – 152) sits immediately after the
training window (days 1 – 122). It measures *one-month* transfer. E022
(depth 12, 1,600 trees) beat E016 by +0.017 PR-AUC on that window and held
the advantage in every 10-day block of it — and then transferred only
+0.004 on the reporting window one month further out, with slightly worse
operating-point metrics (`docs/xgboost_progression.md`, model card). The
adjacent window could not see the second month. That observation came
from a test-window consultation (ADR 0002's log, #5); the remedy below
uses **only days ≤ 152** and is adopted *before* any further selection.

## Options

1. **Keep the single adjacent window.** Cheapest; blind to multi-month
   decay by construction.
2. **A single gapped window** (train 1–92, validate 123–152). Sees one
   horizon, loses a month of training data, still one number.
3. **Rolling temporal backtests with explicit horizons.** Several training
   cut-offs; each scored on later 30-day windows at several gaps. Costs 3
   fits per candidate instead of 1; yields a horizon curve instead of a
   point.
4. **Expanding-window CV with a fixed gap** (as in tuning) — a special case
   of 3 with one horizon.

## Decision

**Option 3.** Development data = days 1 – 152 (unchanged; the reporting
window 153 – 183 is untouched by this protocol).

| training cut-off *T* | scored windows `[T + gap + 1, T + gap + 30]` for gap ∈ … |
|---|---|
| 60 | 0, 7, 14, 30, 60 |
| 90 | 0, 7, 14, 30 |
| 120 | 0 |

Ten (fit, window) evaluations from three fits per candidate. A window is
used only if it ends by day 152. All fitted state (frequency tables,
one-hot vocabularies, the model) is learned on days ≤ *T*; calibration is
not part of the backtest (it is fit afterwards on the shipped candidate as
in ADR 0007).

**Selection criterion.** A candidate is compared to the current best on
**mean validation PR-AUC over the horizons with gap ≥ 30 days** (four
windows: T=60 gaps 30 & 60, T=90 gap 30 — three, plus none at T=120; see
consequences), with the **adjacent horizon (gap 0) reported alongside** and
the **decay** (gap-0 mean minus gap-≥30 mean) reported as the robustness
number. The ADR 0003 seed rule applies to the ≥ 30-day mean. A candidate
that wins the adjacent horizon and loses the ≥ 30-day horizons is not
accepted.

## Consequences

- `configs/backtest.yaml`, `fraud.evaluate.backtest`, `scripts/backtest.py`;
  MLflow experiment `fraud-backtest`.
- Three fits per candidate: for the shipped configuration ≈ 10 minutes on
  the development machine. Acceptable for candidate comparison, not for
  every sweep point; sweeps stay on the adjacent window and their winners
  are confirmed here.
- Only three windows exist at gap ≥ 30 within days ≤ 152; the estimate is
  coarser than the adjacent one and the seed rule matters more. If the
  project ever re-freezes its reporting window, extending the development
  span would add horizons.
- ADR 0002's single adjacent window remains the *reporting* validation
  window for the results tables; this protocol is the *selection* protocol
  from now on.
- The first application (E023) re-compares E016 and E022 under the new
  protocol. Whichever wins is the shipped model; the reporting-window
  numbers are then reported with ADR 0002's caveat.
