# Retraining: the offline lifecycle, and the scheduled cycle

Two things live here. **The lifecycle** (`scripts/retrain.py`) replays several
calendar cut-offs offline to show every step works and to measure what
staleness costs — that is the table below. **The scheduled cycle**
(`scripts/retrain_cycle.py`, ADR 0012) runs *one* cut-off against the artifact
that is actually serving, and is what `.github/workflows/retrain.yml` calls
monthly: decide → retrain → gate → promote → publish → open the policy pull
request. Merging that request deploys (`deploy-champion.yml`).

The one step that is not automated is reading that diff. It carries the block
and review bands the new champion re-derived, and they decide what happens to a
customer's transaction; the service takes them from a reviewed commit and not
from a fetched manifest (`docs/security.md`), so a machine may propose them and
a person approves them.

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

The table below assumes labels are final on the day (`label_maturity_days:
0`). They are not: a legitimate label is only the absence of a report
within 120 days (ADR 0009). `--label-maturity-days D` makes the cycle use
labels through T − D and scores the month each artifact actually served
once those labels mature; `docs/feedback.md` has that run and the
staleness it implies.

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

## The scheduled cycle (ADR 0012)

```
label feed clock (+ the monitor's eventual PR-AUC, docs/monitoring.md)
  → fraud.train.trigger.decide     run a cycle, or record why not
  → fraud.train.cycle.run_cycle    challenger fit on days ≤ train_end;
                                   challenger AND the serving champion scored on
                                   train_end+1 … mature_through, which neither saw;
                                   ADR 0010's gates + the cycle's margin
  → promote                        models/champion/ + registry alias (docs/promotion.md)
  → publish                        one immutable champion-<sha12> release
  → pull request                   configs/serving.yaml: bands + champion_sha256
  → merge → deploy-champion.yml    point the host at it, redeploy, verify the digest
```

**The evaluation window moves with the model.** `scripts/promote.py` measures a
candidate on the frozen validation window (days 123–152, ADR 0002). That is the
right window for a model trained on the frozen training window and the wrong one
for a model retrained *through* it — those rows are now training data. So the
cycle scores both models on the latest month of matured labels, one day after
the challenger's training data ends, and **re-scores the champion there**
rather than reusing its manifest metrics, which belong to a different window.
If the champion was itself fitted inside that month, the cycle raises instead of
reporting a comparison that would flatter it.

**What starts a cycle** (`configs/retrain.yaml` → `cycle:`):

| rule | value | why |
|---|---|---|
| `min_new_train_days` | 30 | a month of matured labels the champion has not seen. Less is churn |
| `monitor_pr_auc_drop` | 0.03 | eventual PR-AUC this far below its reference brings a cycle forward — but only when new training days exist, because refitting the same rows cannot answer drift |
| `promotion_margin` | 0.005 | on top of every ADR 0010 gate. Those gates accept a candidate 0.005 *worse* than the champion, which is right for a person choosing a model and wrong for a job replacing one in service |
| `label_maturity_days` | 120 | the host's rule (ADR 0009). The offline lifecycle above uses 0 to reproduce its published table; a scheduled cycle must not |
| `allow_reporting_window` | false | a cycle whose month reaches day 153 consumes held-out data. Turning this on is a consultation to log in ADR 0002 |

**What this dataset lets the automation do is narrow, and the pipeline says so.**
Under a 120-day maturity, 183 days of data leave no month the current champion
(trained through day 122) was not itself fitted on: a scheduled cycle stops with
"labels are mature only through day 63" or "the champion was trained through day
122, inside the evaluation month". That is the delayed-label cost
`docs/feedback.md` measured, stated by the code rather than by a paragraph.
Exercising the machinery needs `--label-maturity-days 0` — the offline
simulation's assumption — or months this dataset does not have.

```bash
uv run python scripts/retrain_cycle.py --dry-run                      # decide, fit, gate, stop
uv run python scripts/retrain_cycle.py --as-of-day 152 --label-maturity-days 0
uv run python scripts/retrain_cycle.py --force --champion-dir /tmp/c  # somewhere harmless
```
