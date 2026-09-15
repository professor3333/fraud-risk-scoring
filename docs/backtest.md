# Rolling temporal backtests (ADR 0008, E023)

`uv run python scripts/backtest.py --candidates …` → `reports/backtest/`.
Development data only (days ≤ 152); the reporting window is never scored.
Three training cut-offs, ten scored windows per candidate:

| train through | scored windows (gap → days) |
|---|---|
| day 60 | 0 → 61–90 · 7 → 68–97 · 14 → 75–104 · 30 → 91–120 · 60 → 121–150 |
| day 90 | 0 → 91–120 · 7 → 98–127 · 14 → 105–134 · 30 → 121–150 |
| day 120 | 0 → 121–150 |

Selection number: **mean PR-AUC over the three windows with gap ≥ 30**
("robust"); the adjacent mean (gap 0) and the decay are reported beside it.

## E023 — E016 vs depth-10 vs E022 (seed 42)

| candidate | adjacent | gap 7 | gap 14 | gap 30 | gap 60 | **robust (≥ 30)** | decay |
|---|---:|---:|---:|---:|---:|---:|---:|
| E016 (depth 8, 800 trees) | 0.6258 | 0.5960 | 0.5668 | 0.5276 | 0.4708 | 0.5087 | 0.117 |
| depth 10, 1,200 trees | 0.6288 | 0.5957 | 0.5675 | 0.5286 | 0.4748 | 0.5107 | 0.118 |
| **E022** (depth 12, 1,600 trees) | **0.6388** | **0.6025** | **0.5750** | **0.5344** | **0.4768** | **0.5152** | 0.124 |

Per window (`reports/backtest/windows.csv`): E022 is the best candidate in
**all ten** windows. Its lead over E016 is +0.013 at gap 0, +0.007 at gap
30 and +0.006 at gap 60 — it halves with distance and does not reverse.
Depth 10 sits between the two everywhere.

## Reading it

1. **The horizon effect is large and monotone for every candidate**: from
   ~0.63 adjacent to ~0.53 one month out and ~0.47 two months out. Part of
   that is genuine drift; part is that the far windows are scored by models
   trained on only 60–90 days. The *comparison* between candidates is on
   identical folds, so it is unaffected; the absolute decay overstates
   pure drift.
2. **E022's advantage is drift-sensitive but not drift-fragile.** The
   reporting window showed its +0.017 adjacent gain becoming +0.004 two
   months out; the backtest shows the same shape inside development data
   (+0.013 → +0.006). So the earlier reading — "more capacity transfers
   worse" — is right about the *slope* and wrong about the *sign*: the
   deeper model still wins further out, by less.
3. **Would the protocol have caught anything?** Yes, in the sense that
   matters: the number the project would have promoted on (+0.0065 robust,
   single seed) is in ADR 0003's seed-pair band rather than a clear pass,
   and the horizon curve makes the shrinking visible before shipping. Under
   the old adjacent-only window, +0.017 looked unambiguous.

## Seed-paired decision

Robust mean (gap ≥ 30), E022 vs E016, seeds 42 / 1 / 2:

| seed | E016 | E022 | Δ |
|---|---:|---:|---:|
| 42 | 0.5087 | 0.5152 | +0.0065 |
| 1 | 0.5113 | 0.5155 | +0.0042 |
| 2 | 0.5096 | 0.5198 | +0.0102 |
| **paired mean** | | | **+0.0070** |

Every pair positive, paired mean above 0.005: **E022 is confirmed under the
drift-aware protocol.** Its edge is about half of what the adjacent window
suggested (+0.013 → +0.007), which is exactly what the reporting window had
hinted at (+0.017 → +0.004) — but it does not reverse, so the shipped model
stands on validation-only evidence rather than on a test-window rescue.
E023's entry in `docs/experiments.md` records the decision.
