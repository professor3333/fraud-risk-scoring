# Tuned versus untuned

The one comparison the stage checklist asked for, on one page. Everything
here is already in MLflow (`fraud-xgboost`, `fraud-tuning`, `fraud-sweeps`)
and told as a story in `docs/xgboost_progression.md`; this is the ledger
view. Validation PR-AUC on the frozen split (ADR 0002); seed noise 0.002 sd
(E004); acceptance ≥ 0.01 at one seed or ≥ 0.005 seed-paired (ADR 0003).

## The same features, untuned and tuned

| | features | parameters | seeds 42 / 1 / 2 | mean | train | gap |
|---|---|---|---|---:|---:|---:|
| **untuned** (E006, `xgb_v2_freq`) | v2 + frequency | defaults: 600 trees, η 0.05, depth 6, mcw 5, subsample 0.8, colsample 0.6 | 0.578 / 0.580 / 0.581 | **0.580** | 0.81 | 0.23 |
| **tuned** (E008, `xgb_v2_tuned`) | v2 + frequency | 16-trial random search, expanding-window CV inside the training window: 800 trees, η 0.05, depth 8, mcw 5, subsample 0.8, colsample 0.5, λ 1, α 1 | 0.615 | **0.615** | 0.92 | 0.30 |

**+0.036 on validation** for the search, against a 0.002 noise floor —
eighteen standard deviations, the largest single step in the project
after the first tree model (E003, +0.17 over logistic regression). Inside
the search's own CV the defaults scored 0.607 and the best trial 0.626,
so the CV gap (+0.019) understated the validation gap; the direction and
the ranking of trials were what the CV was for, and both held.

What the search bought was capacity: every depth-4 trial scored below the
defaults, depth 8 filled the top three, and the learning curve was still
rising at 800 trees. The gap grew from 0.23 to 0.30 — the cost is
recorded, and E021 later showed it is account memorisation, not a metric
that turns over (`docs/xgboost_progression.md` § *What the arc looks like
here*).

## The second round, on the shipped features

| | features | parameters | seeds 42 / 1 / 2 | mean | paired Δ | train | gap |
|---|---|---|---|---:|---:|---:|---:|
| E008 parameters (E016, `xgb_f5_interactions`) | f5 | as tuned above | 0.619 / 0.622 / 0.621 | 0.621 | — | 0.93 | 0.31 |
| one-axis sweeps (E021) | f5 | 17 single-parameter moves from E016 | best single move: depth 12, 0.629 | | | 0.99 | 0.36 |
| **capacity** (E022, `xgb_f5_capacity`, shipped) | f5 | depth 12, mcw 1, 1,600 trees, η 0.05, subsample 0.8, colsample 0.5, λ 1, α 1 | 0.637 / 0.636 / 0.640 | **0.638** | **+0.017**, every seed + | 1.00 | 0.36 |

The second round is not a search but a direction read off the sweeps —
"more capacity" — then seed-paired against the incumbent. +0.017 paired
clears the rule three times over. Its transfer to the reporting window was
+0.004 (E016 0.557 → E022 0.561), which is what prompted the drift-aware
validation protocol (ADR 0008); under that protocol E022 still wins all
ten horizon windows, by +0.013 adjacent and +0.006 at 60 days (E023).

## What tuning did not do

- It never found a turn-over. Every regulariser tried (mcw 20 / 100, λ 10 /
  100, η 0.02) lowered train *and* validation; the "overfit" corner (train
  1.000) scored 0.627. The train/validation gap here measures how well the
  trees memorise the training window's accounts, which the temporal split
  then cannot use — see `docs/xgboost_progression.md`.
- It moved the top of the ranking less than the middle. Recall at
  precision ≥ 0.90 went 0.294 (untuned) → 0.330 (E008) → 0.334 (E022):
  the first round bought most of it, the second almost none, while PR-AUC
  kept rising; block precision at the 0.42 bar is 0.80 for both E016 and
  E022 on validation.
- It cost fit time: about 4 minutes for E022 (`fit_seconds` 259 in
  MLflow) versus well under a minute for the defaults on 8 cores, and the
  same 1.5 GB image either way.

## Reproduce

```bash
uv run python scripts/train.py --model configs/model/xgboost_v2_freq.yaml      # untuned, ~1 min
uv run python scripts/tune.py --config configs/tuning/xgboost.yaml             # E008 search, ~1 h
uv run python scripts/train.py --model configs/model/xgboost_v2_tuned.yaml     # tuned
uv run python scripts/param_sweep.py --config configs/tuning/sweep.yaml        # E021, ~1 h
uv run python scripts/train.py --model configs/model/xgboost_f5_capacity.yaml  # E022, ~4 min
```
