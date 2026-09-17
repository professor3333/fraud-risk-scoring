# Deployment

The service is packaged as a Docker image (`Dockerfile`: runtime deps from
`uv.lock`, non-root user, healthcheck, the champion directory and its frozen
sample). It runs anywhere Docker runs. The public target had to satisfy
**zero payment, no card**; the options were measured, not assumed:

| host | free tier | verdict |
|---|---|---|
| Fly.io | none; the 1 GB machine is paid, card required | paid alternative, config kept |
| Hugging Face Spaces (Docker) | Docker Spaces now require PRO (`402 Payment Required` on create) | out |
| Google Cloud Run, Oracle, AWS, Azure | free quotas, but a card to open the account | out |
| **Render** free web service | **512 MB, 0.1 CPU, Docker, no card**, sleeps after 15 min idle | **default target** |
| Koyeb free | 512 MB, 0.1 vCPU, no card | equivalent; same image works |

The service fits: **238 MiB at rest, 284 MiB after a 5,000-row upload,
316 MiB after several** (the 538 MiB in README → Performance is twenty
concurrent uploads). At a tenth of a CPU, measured locally with
`--cpus 0.1 --memory 512m`: cold start 62 s, the 200-row sample scores in
19 s, a single prediction in 1.2 s, `deploy_check.py` passes. A demo, not
a service; the README says so next to the link.

## The hosted image

```
public repo ──(Render builds deploy/hosted/Dockerfile)──► image WITHOUT weights
champion-<sha> GitHub release (models/champion/ as assets) ──► fetch_champion() at startup ──► parity check ──► serve
```

- `deploy/hosted/Dockerfile` is the root Dockerfile minus the `COPY
  models/champion/` step, binding to `$PORT`. `render.yaml` is the
  blueprint: one free Docker web service, health check on `/health`,
  `autoDeploy: false` so only the release workflow deploys.
- The weights never enter git or the image. At startup
  `fraud.serve.app.fetch_champion` downloads `model.joblib`, the frozen
  golden, the manifest and the monitoring reference from
  `FRAUD_CHAMPION_URL` — the assets of a `champion-<sha12>` pre-release on
  this repository (`scripts/publish_champion.py`; one immutable release per
  promoted champion, public, no token, free) — and the usual startup parity
  check runs: a tampered or mismatched file and the service never becomes
  healthy. The URL pins the served weights by content hash, so what a
  running service serves is readable from its environment. A store that
  needs a bearer token takes it from `FRAUD_CHAMPION_TOKEN`. Publishing the
  weights is a choice: this is a portfolio model on public Kaggle data and
  every metric is already in the README, so a private store bought nothing
  but an extra account.
- `FRAUD_API_KEY` is **not** set on the demo: a dashboard that needs a key
  is not a demo. The upload cap, the 60 s time budget and the instance's
  own sleep are the protections; the audit trail is ephemeral on the free
  plan, so `/outcomes` writes nothing that outlives a restart. On 0.1 CPU
  uploads much beyond the 200-row sample run into the time budget.

## One-time setup (needs the account owner, once; free — done 2026-09-17: https://fraud-risk-scoring-m1fp.onrender.com)

```bash
uv run python scripts/publish_champion.py
#   → creates the champion-<sha12> pre-release with models/champion/ as assets,
#     prints FRAUD_CHAMPION_URL (already done for the current champion:
#     https://github.com/professor3333/fraud-risk-scoring/releases/download/champion-7af85ec92813)
```

Then on https://dashboard.render.com (sign up with GitHub, no card):
*New → Blueprint*, pick this repository — `render.yaml` creates the
service — and set its one environment variable `FRAUD_CHAMPION_URL`. The
first build starts on its own; wait for `/health`, then:

```bash
uv run --no-project python scripts/deploy_check.py https://fraud-risk-scoring-m1fp.onrender.com
# continuous deployment on every tag: the service's Settings → Deploy Hook
gh variable set RENDER_URL --body https://fraud-risk-scoring-m1fp.onrender.com
gh secret set RENDER_DEPLOY_HOOK --body '<the hook URL>'
```

From then on `uv run python scripts/release.py vX.Y.Z` cuts a release
whose tag posts the deploy hook, waits until the served version is the
tag's, runs `deploy_check.py` against the live URL and cuts the GitHub
release. The live URL goes into the README's *Live demo* line.

## Fly.io (alternative)

```bash
brew install flyctl
flyctl auth login
flyctl apps create fraud-risk-scoring
flyctl secrets set FRAUD_API_KEY=…            # optional
flyctl volumes create fraud_audit --size 1    # persistent audit trail (mounted at /app/audit)
uv run python scripts/release.py vX.Y.Z --fly-image   # builds + pushes registry.fly.io/<app>:vX.Y.Z
gh variable set FLY_APP --body fraud-risk-scoring && gh secret set FLY_API_TOKEN --body "$(flyctl tokens create deploy -x 999999h)"
```

## The check every deployment must pass

`scripts/deploy_check.py <url>` asserts: `/health` is `ok` with
`parity_rows > 0` (the startup parity check ran), `/model-info` reports
the same version, and the synthetic sample CSV scores 200 rows with the
same model version. If the artifact does not reproduce its frozen golden
the service never becomes healthy and the check fails — the intended
behaviour. It sends `X-API-Key` when `/health` says the service is keyed.

## Updating the model

1. Train, calibrate, freeze (`README.md` → *Reproduce the model*).
2. `uv run python scripts/promote.py --run-name <run>` — the acceptance
   gates (`docs/promotion.md`); on success `models/champion/` is replaced
   and the registry alias moves. Nothing in `configs/`, the `Dockerfile` or
   `.dockerignore` changes.
3. `uv run python scripts/publish_champion.py --repo <user>/<model-repo>`
   so the hosted service can fetch it, then release (below).

To revert, promote the previous champion again; it goes through the same
gates.

## Release and continuous deployment

```
scripts/release.py vX.Y.Z   (the machine that holds models/champion)
  ├─ preconditions: main, clean tree, pyproject version == X.Y.Z, tag unused,
  │                 champion reproduces its golden
  ├─ tests → ruff → format → mypy                      (§14 ship sequence)
  ├─ [--fly-image] flyctl deploy --build-only --push --image-label vX.Y.Z
  └─ git tag -a vX.Y.Z && git push origin vX.Y.Z
                │
                ▼  .github/workflows/deploy.yml (on the tag)
  checks   ci.yml (lint, types, tests, fixture image, container health)
  render   POST the deploy hook → Render builds main (== the tag) from deploy/hosted/Dockerfile,
           the service fetches the champion at startup → wait for the tag's version
           → scripts/deploy_check.py https://<service>.onrender.com
  fly      flyctl deploy --image registry.fly.io/<app>:vX.Y.Z → deploy_check.py
  release  gh release create vX.Y.Z --generate-notes
```

The weights are git-ignored and are not in any image a public runner
builds. On the Render leg the host builds an image without them and the
service fetches the champion from its GitHub release at startup;
on the Fly leg the runner deploys an image built where the champion is.
A leg whose variables are unset is skipped; the release is still cut.
`scripts/release.py --dry-run` prints the sequence without doing it.

## Hardening for a public URL

| protection | where | behaviour |
|---|---|---|
| API key | `FRAUD_API_KEY` (an environment variable on the host; unset on the free demo) | when set, `/predict*`, `/explain`, `/outcomes` and `/audit/*` require a matching `X-API-Key` (constant-time compare) → 401 before the body is read; `/health`, `/model-info` and the dashboard stay open, and `/health` reports `auth: api_key`. The dashboard shows a key field when the server asks for one and keeps it in the browser only. Unset = open, for local use. |
| upload cap | `serving.yaml` `max_upload_bytes` (25 MB) | declared length checked, then the stream abandoned the moment it exceeds the cap → 413; the row limit stops the parser one row past 5,000 → 422 |
| time budget | `serving.yaml` `request_timeout_s` (60 s) | a guarded call past the budget → 504. It bounds the client's wait; a scoring call already running in the thread pool finishes on its own (`/health` stays responsive, as the check below shows). |
| request IDs | `X-Request-ID` honoured or generated | echoed on every response, stored on every audit row and request record, present in every log line |
| structured logs | logger `fraud.serve.requests` | one JSON line per guarded call — `ts, request_id, method, path, status, latency_ms, rows, client` — success, 401, 422 and 504 alike; Fly ships stdout to `flyctl logs` |
| rate limiting | `fly.toml` `[http_service.concurrency]` soft 20 / hard 50 | Fly's proxy queues then refuses beyond the hard limit per machine; the benchmark (README → Performance) is why 20 is the soft limit |

Checked against a live server with a 0.5 s budget: a 4.7 MB upload without
a key → 401 in 0.9 ms (nothing read); with the key → 504; `/health` 200
in 1.6 ms afterwards.

## Sizing evidence

`fly.toml` asks for 1 GB and a soft / hard request concurrency of 20 / 50.
`scripts/benchmark_api.py --url …` against the image under `--cpus 1
--memory 1g` (README → Performance, `reports/benchmark/docker_1cpu_1gb.md`):
peak memory 538 MiB, cold start 3.5 s, 21 single predictions/s at 20
concurrent clients with p50 0.9 s. Run it against the Fly URL after a
deploy to see what the shared CPU actually gives.

## Local alternative

```bash
docker build -t fraud-risk-scoring . && docker run --rm -p 8000:8000 fraud-risk-scoring
```

Identical image, identical startup check.
