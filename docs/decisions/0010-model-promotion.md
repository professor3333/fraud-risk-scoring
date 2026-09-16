# ADR 0010 — Model promotion: candidate → acceptance gates → champion

**Date:** 2026-09-16 · **Status:** accepted

## Context

Experiments are tracked (MLflow), retraining is simulated with a
champion/challenger rule (`docs/retraining.md`), the service refuses to
start on a parity mismatch — and yet putting a new model into service
meant editing `configs/serving.yaml`, the `Dockerfile` and `.dockerignore`
to the new artifact's file name and copying its validation numbers into
`model_info` by hand (`docs/deployment.md`, PR #27). There was no
"champion" object anywhere, only whichever file name the config happened
to hold; nothing checked that the artifact behind that name cleared the
bars the project had already written down (the 80 % block-precision bar,
the calibration requirement, the cost model, the review budget) before it
was served.

## Options

1. **Keep the manual edit.** One `sed` per promotion; the checks live in
   the engineer's head and the PR review.
2. **Load the model from the MLflow registry at startup** (`models:/name@
   champion`). Removes the file name from the config, but the serving
   image then needs MLflow and the tracking store at runtime, and the
   startup parity check would have to fetch the golden from the same place.
3. **A gate script that materialises the champion into a fixed directory
   and records the decision in the registry.** The service loads a fixed
   path and reads a manifest next to it; the registry is the ledger of
   every candidate and of which version is the champion; the runtime image
   stays MLflow-free.

## Decision

**Option 3.**

- `configs/promotion.yaml` names the gates. A candidate is measured on the
  validation window **under its own re-derived policy** (block threshold at
  the precision bar, review band sized to the 200/day budget), never under
  the champion's thresholds:

  | gate | rule |
  |---|---|
  | PR-AUC | ≥ champion − 0.005 (two seed standard deviations, ADR 0003) |
  | recall at 200 reviews/day | ≥ champion − 0.01 and ≥ 0.60 |
  | ECE | ≤ 0.01 |
  | Brier | ≤ champion + 0.0005 |
  | block precision | ≥ 0.75 (the bar is 0.80 on the grid) |
  | cost per transaction (ADR 0006 costs + review cost) | ≤ champion + 0.02 |
  | parity | the candidate reproduces its frozen golden |

  Absolute bounds always apply; the champion comparison applies when there
  is a champion (the first promotion bootstraps). **Every gate must pass.**
- `scripts/promote.py --run-name <run>` evaluates, registers a version in
  the MLflow model registry (`fraud-risk-scorer`; tags carry the verdict
  and every gate's value) and, if all pass, replaces `models/champion/`:
  `model.joblib`, its frozen golden, a freshly built monitoring reference,
  and `model_manifest.json`. `--dry-run` evaluates and registers without
  promoting. A rejected candidate exits non-zero.
- The service loads `models/champion/model.joblib`. When the manifest is
  present, `model_version`, `model_info` and the policy `bands` come from
  it, and a manifest that does not describe the artifact next to it is
  refused at startup. `serving.yaml`, the `Dockerfile` and `.dockerignore`
  no longer name an artifact.
- Gate tolerances are safety bars, not a selection rule. Choosing between
  candidates stays with the experiment protocol (ADR 0003, ADR 0008); the
  retraining lifecycle's +0.005 promotion margin is that selection rule
  for challengers. The gates exist so that whatever was chosen cannot be
  served without clearing the bars the project already committed to.

## Consequences

- `fraud.train.promotion`, `configs/promotion.yaml`, `scripts/promote.py`,
  `docs/promotion.md`; `reports/promotion/<run>.json` per evaluation.
- Bootstrapped with E022 (`xgb_f5_capacity`, registry version 1). The
  re-derived bands are identical to the ones previously hand-copied into
  `serving.yaml` (review 0.062, block 0.42), which is the check that the
  script measures what the policy documents describe. E016
  (`xgb_f5_interactions`) run as a dry-run candidate is rejected on
  PR-AUC, Brier and cost (registry version 2) — the gates reproduce the
  ranking in `docs/xgboost_progression.md`.
- Updating the model is now: train → calibrate → freeze → `promote` →
  `flyctl deploy`. The Docker layer copies a directory whose name never
  changes.
- Eventual (served-month) performance cannot be a gate at promotion time
  (ADR 0009); it is the number the monitor reports later and the reason a
  promotion may need reverting. Reverting is `promote --run-name
  <previous>`, which re-runs the gates.
- Not done: a gate on the drift-aware backtests (ADR 0008; three fits per
  candidate, run by the experiment protocol instead); automatic rollback
  from the monitor; a registry that stores the artifact bytes (the
  registry records the path and sha, the file lives in `models/`).
