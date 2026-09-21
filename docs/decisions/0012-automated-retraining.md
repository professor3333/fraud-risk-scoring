# ADR 0012 — Automated retraining, promotion and deployment

**Date:** 2026-09-21 · **Status:** accepted

## Context

Every piece of the lifecycle exists and is tested: matured labels
(ADR 0009), a retraining cycle (`docs/retraining.md`), acceptance gates and a
registry (ADR 0010), an immutable champion release, a host that fetches it and
refuses to start unless it reproduces its golden, and since ADR 0011 a
scheduled monitor that says when the served model is degrading.

They are not connected. `docs/retraining.md` says so in three words — *"Not
scheduled"* — and names what is missing: a label feed, a trigger, deployment,
registry integration.

The reason to connect them is measured, not aesthetic. This model decays
faster than any feature work improved it:

| evidence | effect |
|---|---|
| validation → reporting window (`docs/model_card.md`) | PR-AUC 0.637 → 0.561 |
| an incumbent one month stale, same month (`docs/retraining.md`) | 0.522 vs 0.647 — **+0.125 for one month of fresher data** |
| every feature and tuning experiment in the project, E003 → E022 | +0.067 |
| waiting 30 days for labels before retraining (`docs/feedback.md`) | a further −0.045 on the month served |

Retraining is the largest single lever this system has, and it is the one
thing a human has to remember to pull.

Two things make automating it delicate rather than routine:

1. **A retrained model breaks the window the gates were written for.** ADR
   0010 measures a candidate on the frozen validation window, days 123–152.
   A model retrained *through* day 152 has been fitted on those rows, so the
   same script would compare its training scores against the champion's honest
   ones and promote on the difference. An automated pipeline that gets this
   wrong promotes confidently and constantly.
2. **The block and review bands are policy, not metadata.** They decide what
   happens to a customer's transaction. The service already refuses to take
   them from a champion fetched over the network, precisely because that
   manifest is not digest-pinned (`docs/security.md`); they come from
   `configs/serving.yaml`, which ships in the image from a reviewed commit.
   A retrained champion re-derives its own block threshold — so "deploy the new
   model" is inescapably also "change the policy".

## Options

1. **Fully unattended: retrain → promote → deploy, no human at any point.**
   The fastest loop, and the one that most resembles a large fraud team's
   pipeline. Here it would mean either trusting policy bands that arrived over
   the network — undoing ADR 0010's and `docs/security.md`'s reasoning — or
   having a job rewrite the reviewed config and push to `main` unattended.
2. **Automate only the trigger.** The monitor opens an issue saying "a month
   of labels has matured; retrain". Cheap, honest, and leaves the expensive,
   error-prone part — the comparison — exactly where it is: in a human's
   memory of which flags to pass.
3. **Automate everything up to the policy change.** A scheduled job decides
   whether a cycle is due, fits the challenger, scores challenger *and*
   champion on a month neither was fitted on, runs the gates plus a margin,
   promotes into the registry and the champion directory, publishes the
   immutable release — and then opens a pull request carrying the two things
   it may not decide alone: the bands, and the digest they belong to.
   Merging that diff deploys.
4. **Approval-gated deployment** (GitHub Environments with required
   reviewers) instead of a pull request. The approval is a button rather than
   a diff: the reviewer sees "deploy?", not "these thresholds, this artifact".

## Decision

**Option 3, with option 4's gate kept as well.**

The pipeline is `scripts/retrain_cycle.py` + `.github/workflows/retrain.yml` +
`.github/workflows/deploy-champion.yml`:

```
label feed clock (+ the monitor's eventual PR-AUC)
  → fraud.train.trigger.decide      run a cycle, or say why not
  → fraud.train.cycle.run_cycle     challenger fit on days ≤ train_end
                                    challenger AND champion scored on the month
                                    ADR 0010 gates + the cycle margin
  → promote                         models/champion/, registry alias
  → publish                         champion-<sha12> release (immutable)
  → pull request                    configs/serving.yaml: bands + champion_sha256
  → merge → deploy-champion.yml     point the host at it, redeploy, verify
```

Four decisions inside it carry the weight:

**The comparison window moves with the model.** The cycle evaluates on
`train_end+1 … mature_through` — the latest month of matured labels, which the
challenger's training data stops one day short of. The champion is **re-scored
on that same month**; its stored manifest metrics are never reused, because
they were measured on a different window. If the champion was itself trained
inside that month, the cycle raises rather than reporting a comparison that
would favour it. This is the check that keeps an automated promotion honest.

**An unattended job must clear a margin, not merely the gates.** ADR 0010's
gates accept a candidate up to 0.005 PR-AUC *worse* than the champion — right
for a human choosing between models, wrong for a robot replacing one in
service. The cycle additionally requires `challenger ≥ champion + 0.005` on the
month. A tie is churn with deployment risk attached.

**Degradation alone never triggers a cycle.** The monitor's eventual PR-AUC
can bring a cycle forward, but only when training days have matured that the
champion has not seen. Refitting the same rows cannot answer drift, and a job
that retrains because a metric fell would do it every day.

**The reporting window is not consumed by a clock.** A cycle whose evaluation
month reaches day 153 is refused unless `cycle.allow_reporting_window` is true
in a reviewed commit — which is a consultation to record in ADR 0002's log. A
scheduled job must not be able to spend the held-out window because the
calendar rolled forward.

`configs/serving.yaml` gains **`champion_sha256`**: the artifact the bands in
that file were measured on. The service checks it when the champion arrives
over the network — the case where the *host* picks the artifact and the *repo*
supplies the policy — and refuses to serve a pairing nobody approved. Before
this, pointing `FRAUD_CHAMPION_URL` at a newer champion silently served a
block threshold no one had reviewed, and every other check still passed.

## Consequences

- **The loop is closed, with exactly one human step, in the place where a
  human belongs.** Nothing in the chain is manual except reading a diff that
  says: this artifact, this block threshold, this review band, measured on
  these rows, against this champion.
- **Model weights are now built on a public runner.** `deploy.yml` and
  `scripts/release.py` used to say the weights "never enter the repository or a
  public runner"; the first half still holds, the second no longer does, and
  those comments are corrected rather than quietly left. What actually protects
  the artifact was never the runner: it is the digest in the release tag, which
  the service pins the download against before it deserializes anything, plus
  the golden it must reproduce, plus `champion_sha256`. The Kaggle data is
  downloaded with repository credentials and is never uploaded as a run
  artifact.
- **What this dataset will actually let the automation do is narrow, and the
  code says so out loud.** Under the host's 120-day maturity rule
  (`configs/retrain.yaml`), 183 days of data leave no month that the current
  champion (trained through day 122) was not itself fitted on, so a scheduled
  cycle stops with "the champion was trained through day 122, inside the
  evaluation month" or "labels are mature only through day 63". That is not a
  bug to work around: it is the delayed-label cost `docs/feedback.md` measured,
  stated by the pipeline rather than by a paragraph. Demonstrating the machinery
  therefore needs either `--label-maturity-days 0` (the offline simulation's
  assumption) or fresh months the dataset does not have.
- **It reproduces the shipped model.** Run at as-of day 152 —
  the project's own split — the cycle lands on validation PR-AUC 0.63670 against
  the champion's published 0.6367 and re-derives bit-identical policy bands,
  while a champion one month stale scores 0.5305 on the same month
  (`docs/retraining.md` → Two cycles, end to end). That the automated path and
  the hand-run path agree to four decimals is the evidence that this is the same
  pipeline and not a second one.
- **Cost.** One cycle is three fits of the production recipe (1,600 trees,
  depth 12) — the challenger and two calibration folds — plus scoring both
  models and rebuilding the monitoring reference. Measured here: 9 min 44 s for
  a cycle training through day 92, and the artifact it produces is 32 MB. The
  workflow allows five hours and runs monthly.
- **The registry gains a version per cycle**, promoted or not, with its gate
  results — so "why is this model serving?" is answerable from the ledger
  rather than from a workflow log that expires.
- **Not done, deliberately:** automatic rollback (the monitor can say a
  champion degraded; choosing to revert is a decision with the same weight as
  promoting, and `promote --run-name <previous>` already exists); training on
  censored labels to shorten the effective lag (`docs/feedback.md` names it as
  the thing that would help most, and it is a modelling change, not an
  orchestration one); searching hyper-parameters inside a cycle (one recipe,
  one seed — a search would need its own time-ordered protocol, ADR 0008);
  and any automatic change to the *review budget*, which is an operations
  decision, not a model one.

## Files

`src/fraud/train/trigger.py`, `src/fraud/train/cycle.py`,
`src/fraud/train/serving_config.py`, `scripts/retrain_cycle.py`,
`configs/retrain.yaml` → `cycle:`, `configs/serving.yaml` →
`champion_sha256`, the startup check in `src/fraud/serve/app.py`,
`.github/workflows/retrain.yml`, `.github/workflows/deploy-champion.yml`,
`tests/test_retrain_cycle.py`, and `docs/retraining.md` → Scheduled cycles.
