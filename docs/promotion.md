# Model promotion

A trained, calibrated, frozen artifact is a **candidate**. It becomes the
**champion** — the model the service loads — only by clearing every
acceptance gate against the current champion on the validation window and
reproducing its frozen golden (ADR 0010).

```
models/<run>_calibrated.joblib + frozen golden            candidate
        │
        ▼  scripts/promote.py --run-name <run>
metrics on validation under the candidate's own policy   (block at the 80 % precision bar,
        │                                                 review band sized to 200/day)
        ▼
gates vs the champion's manifest (configs/promotion.yaml) + parity
        │
        ├─ registry: fraud-risk-scorer version N, tags = verdict + every gate   (always)
        │
        └─ all pass ──► models/champion/   model.joblib · model_frozen_{sample,expected}.json
                                           model_monitor_reference.json · model_manifest.json
                        alias champion ──► version N
        │
        ▼
service: loads models/champion/model.joblib; version, facts and bands from the manifest
```

## Gates

| gate | rule | why this bar |
|---|---|---|
| `pr_auc` | ≥ champion − 0.005 | two seed standard deviations (ADR 0003) |
| `recall_at_200_per_day` | ≥ champion − 0.01, ≥ 0.60 | the analyst budget the review policy is sized to |
| `ece` | ≤ 0.01 | probabilities are used, not only ranks (ADR 0007) |
| `brier` | ≤ champion + 0.0005 | calibration must not regress |
| `block_precision` | ≥ 0.75 | the block band is defined by the 0.80 bar; 0.75 allows grid rounding |
| `cost_per_transaction` | ≤ champion + 0.02 | ADR 0006 costs plus the review cost, per transaction |
| parity | golden reproduced exactly | G8 |

Absolute bounds always apply. The champion comparison applies when there is
a champion; the first promotion is a bootstrap. Every gate must pass; one
failure rejects (exit status 1) and the registry records the rejection.

Tolerances are safety bars, not a selection rule: they say "not worse than
what is served, by more than noise". Which candidate to put forward is the
experiment protocol's job (ADR 0003, ADR 0008), and the retraining
lifecycle's +0.005 margin is its own selection rule for challengers.

## What happened

| candidate | PR-AUC | recall@200/d | ECE | Brier | block precision | cost/txn | verdict | registry |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `xgb_f5_capacity` (E022, shipped) | 0.637 | 0.734 | 0.0037 | 0.0185 | 0.800 | 1.848 | **promoted** (bootstrap) | v1, `champion` |
| `xgb_f5_interactions` (E016, dry run) | 0.619 | 0.725 | 0.0049 | 0.0191 | 0.800 | 1.890 | rejected: PR-AUC, Brier, cost | v2 |

The bootstrap re-derived the policy bands from the validation window —
review 0.062, block 0.42 — and they are identical to the values that had
been copied into `serving.yaml` by hand from `docs/review_policy.md`. That
is the check that `promote.py` measures what the policy documents
describe. E016's rejection reproduces the E016 → E022 comparison in
`docs/xgboost_progression.md` by three independent gates.

## The champion directory

`models/champion/` is written only by `promote.py` and is what the
`Dockerfile` copies. The manifest carries: run name, artifact sha256, the
source artifact, registry version, promotion time, the bands, every metric
and every gate result, and the `model_info` the API reports. At startup
the service refuses an artifact whose sha does not match its manifest, and
one that does not reproduce its golden. `serving.yaml` still holds
`model_version`, `model_info` and `bands` — as fallbacks for a champion
directory without a manifest, which is what CI's fixture stand-in is.

## Operating it

```bash
uv run python scripts/freeze_artifact.py --run-name <run>       # a candidate needs a golden
uv run python scripts/promote.py --run-name <run> --dry-run     # gates + registry, no change
uv run python scripts/promote.py --run-name <run>               # promote if every gate passes
uv run python scripts/publish_champion.py                       # champion-<sha12> release
#   → then set FRAUD_CHAMPION_URL on the host to that release (see below)
uv run python scripts/release.py vX.Y.Z --full-checks           # tag → deploy → verify
uv run python scripts/promote.py --run-name <previous>          # revert = promote the previous champion
```

`reports/promotion/<run>.json` has every evaluation; `mlflow ui` shows the
registry with the `champion` alias and each version's gate tags.

## Shipping a promoted champion

Promotion writes `models/champion/`. That is the *local* champion. The hosted
service has no access to it: its image carries no weights and it fetches the
champion at startup from whatever `FRAUD_CHAMPION_URL` is set on the host
(`docs/deployment.md`). Three things must line up:

```
promote.py        → models/champion/            (local: the promoted champion)
publish_champion.py → champion-<sha12> release  (public: the fetchable copy)
FRAUD_CHAMPION_URL on the host → that release   (manual: what is actually served)
```

**The third step is manual and nothing infers it.** Promoting and publishing a
new champion does not change what the service serves; the host keeps fetching the
release its environment names. That failure is silent by construction: the old
champion is correctly digest-pinned to its own release, so it fetches, verifies
and serves cleanly, and `/health` is green throughout. The only tell is the sha
in `model_version`.

It is manual because the URL *is* the trust anchor. The digest that `model.joblib`
is checked against before it is deserialized comes from the `champion-<sha12>` tag
in that URL, deliberately from deployer configuration rather than from anything
the artifact store serves (ADR: `docs/deployment.md`). Resolving "the latest
champion" at startup would automate the step by removing the anchor, which is the
one thing it must not do. Automating it properly means the release pipeline
setting the URL through the host's API — the authority moves to CD, the pin stays
out-of-band. That is not wired up; the two guards below are what stand in for it.

**A promotion that changes the bands needs a config commit.** The hosted service
refuses to take `bands` from a fetched manifest — they decide block and review, and
the manifest is not digest-pinned the way `model.joblib` is (`docs/security.md`) —
so it reads them from `configs/serving.yaml` and **fails startup if the champion's
manifest disagrees**. Promoting a champion with new thresholds therefore means
updating that config in the same release. The failure is loud and names both values;
it cannot be forgotten into serving the wrong policy.

Two checks keep the mismatch from reaching a release unnoticed:

- `scripts/release.py` refuses to tag when the promoted champion has no
  `champion-<sha12>` release, and records the champion's sha in the tag
  annotation.
- `deploy.yml` reads that sha back and fails the release unless the live
  `/health` reports the service is serving that same champion.

Neither can update the host for you. They convert a silent staleness into a
failed release that names the URL to set.

## What it does not do

- No gate on eventual (served-month) performance — it cannot be known at
  promotion time (ADR 0009). That number arrives from the monitor months
  later and is the reason a promotion may be reverted.
- No gate on the drift-aware backtests (ADR 0008): three fits per
  candidate, run by the experiment protocol before a candidate is put
  forward.
- No automatic rollback; no artifact bytes in the registry (it records the
  path and sha; the file lives in `models/`).
