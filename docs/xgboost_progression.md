# XGBoost progression: defaults → search → one-axis sweeps

Evidence: MLflow `fraud-xgboost`, `fraud-tuning`, `fraud-sweeps`;
`reports/tuning/`, `reports/sweeps/`, `reports/curves/`. Validation PR-AUC on
the frozen split throughout; seed noise 0.002 sd.

## V1 — sensible defaults (E003)

600 trees, η 0.05, depth 6, min_child_weight 5, subsample 0.8, colsample 0.6,
λ 1, on the raw columns: **0.570** (logistic regression 0.402). Train 0.79,
gap 0.22. Accepted as the first tree model; the gap was recorded, not acted
on.

## V2 — a small time-ordered search (E008)

16 random trials with expanding-window CV inside the training window, the
defaults scored as a reference on the same folds. Reference CV 0.607; best
trial 0.626 (800 trees, depth 8, mcw 5, subsample 0.8, colsample 0.5, λ 1,
α 1). Refit on the full training window: **0.616** (+0.038 on validation).
Every depth-4 trial scored below the reference; depth 8 filled the top
three. Gap 0.30; the learning curve showed validation still rising at 800
trees.

## V3 — one parameter at a time (E021)

Anchored on the shipped model (E016 features, E008 parameters, 0.619).
Each row changes one thing (`reports/sweeps/one_at_a_time.csv`):

| axis | setting | train | **val** | gap | recall @ P ≥ 0.90 |
|---|---:|---:|---:|---:|---:|
| — | anchor (depth 8, η 0.05, mcw 5, 800 trees) | 0.927 | **0.619** | 0.31 | 0.329 |
| depth | 3 | 0.664 | 0.518 | 0.15 | 0.216 |
| depth | 6 | 0.846 | 0.592 | 0.25 | 0.298 |
| depth | 12 | 0.989 | **0.629** | 0.36 | 0.338 |
| η | 0.02 | 0.836 | 0.592 | 0.24 | 0.289 |
| η | 0.15 | 0.994 | 0.618 | 0.38 | 0.347 |
| min_child_weight | 1 | 0.957 | **0.624** | 0.33 | 0.314 |
| min_child_weight | 20 | 0.867 | 0.599 | 0.27 | 0.304 |
| min_child_weight | 100 | 0.766 | 0.556 | 0.21 | 0.259 |
| subsample | 0.5 | 0.921 | 0.610 | 0.31 | 0.304 |
| subsample | 1.0 | 0.920 | 0.616 | 0.30 | 0.323 |
| colsample | 0.2 | 0.909 | 0.604 | 0.31 | 0.309 |
| colsample | 1.0 | 0.939 | 0.621 | 0.32 | 0.320 |
| α (L1) | 0 | 0.919 | 0.613 | 0.31 | 0.316 |
| α (L1) | 10 | 0.905 | 0.617 | 0.29 | 0.306 |
| λ (L2) | 10 | 0.900 | 0.611 | 0.29 | 0.315 |
| λ (L2) | 100 | 0.837 | 0.590 | 0.25 | 0.276 |
| **"overfit"**: depth 12, mcw 1, no subsampling, η 0.15 | | **1.000** | **0.627** | 0.37 | 0.366 |
| "regularised": depth 12, mcw 20, subsample 0.8, colsample 0.5, η 0.05, α 1, λ 10 | | 0.921 | 0.616 | 0.31 | 0.320 |

Trees, read post hoc from one 2,000-tree fit at η 0.05
(`reports/sweeps/trees.csv`): validation 0.535 at 100 → 0.602 at 500 →
0.619 at 800 → 0.622 at 1,000 → 0.626 at 1,700 → 0.626 at 2,000, while
train climbs to 0.99. It flattens; it does not turn down.

## What the arc actually looks like here

The textbook sequence is *underfit → add capacity → overfit → regularise*.
The left half is there: depth 3, η 0.02, min_child_weight 100 and λ 100
all underfit — low train **and** low validation. The right half is not:
within the ranges tested, **validation never turns over**. Depth 12 beats
depth 8; min_child_weight 1 beats 5; 1,700 trees beat 800; the model that
memorises the training window perfectly (train PR-AUC 1.000) is *better* on
validation than the anchor, and putting regularisation back on it costs
0.011. Every regulariser tried moved validation down or nowhere.

Why the gap misleads on this data:

1. **Part of the gap is account memorisation, not overfitting.** Labels
   propagate within an account (ADR 0001), and the training window contains
   many rows per account. A deep tree that isolates an account's rows gets
   them all "right" in training. That inflates train PR-AUC toward 1.0
   without harming the ranking of *new* rows — the same mechanism that made
   the random split report 0.81 (E010). The train score is not measuring
   generalisation; only the validation curve is.
2. **The signal is high-order interactions among many weak columns**
   (`docs/ablation.md`, `docs/error_analysis.md`: the fraud archetype is a
   conjunction of six or seven fields). Depth is what expresses
   conjunctions; shallow trees with strong regularisation cannot represent
   them and underfit.
3. **420k rows and 3.5 % positives** is enough data that the variance
   penalty of deep trees is small relative to the bias penalty of shallow
   ones — at least up to depth 12 and 2,000 trees. The gap widens because
   the training fit improves, not because validation degrades.

The lesson recorded: **the train/validation gap is a symptom to look at,
not a metric to minimise.** The decision criterion stays validation PR-AUC
under the seed-paired rule; the gap and the learning curve are there to
catch the *turn*, and the turn has not arrived in the tested ranges. What
would change this picture is a larger test of drift: a model that memorises
more may transfer worse a month out. That is measured, not assumed — E022
takes the sweep's direction as a candidate and, if accepted, gets its single
test look like every other shipped model.

## V4 — the sweep's direction, seed-paired and shipped (E022)

Depth 12, min_child_weight 1, 1,600 trees at η 0.05: **0.637 / 0.636 /
0.640** on seeds 42 / 1 / 2 versus E016's 0.619 / 0.622 / 0.621 — paired
+0.017, positive on every seed, and the advantage holds across the whole
validation month (+0.020 / +0.014 / +0.018 by 10-day block). Accepted and
shipped as v0.3.0.

**And then the test look.** +0.017 on validation became **+0.004** on the
test month (0.561 vs 0.557), with the operating-point numbers slightly
*worse* (precision at 0.08 0.262 vs 0.275; cost 275k vs 262k; recall at 500
reviews/day 0.801 vs 0.816). This is the drift caveat above, now measured:
the model that memorises the training window harder transfers a little
worse in the second month. Validation could not see it because the
validation window sits immediately after training. The rule was applied as
written — validation decides, test reports — and the result is recorded as
what it is: evidence that the *validation horizon*, not the model, is the
next thing to fix. A drift-aware protocol (a gap between training and
validation, or a validation window as far out as deployment will be) is
the follow-up, to be decided in an ADR before any further model selection.
Re-deciding E022 on the test number would make every later test number
optimistic; that trade is not taken.
