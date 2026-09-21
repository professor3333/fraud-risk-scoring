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

**Public demo persistence is ephemeral; durable auditing/feedback requires persistent storage.**
The free Render deployment loses its local SQLite state on restart or redeploy:
prediction audit records, any stored delayed labels, cross-request review-budget
accounting, and locally stored monitoring history. Local Docker and Fly deployments
can retain this state using their configured persistent SQLite volumes. Durable
storage is intentionally outside the scope of the free portfolio demo.

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
  promoted champion, public, no token, free). `model.joblib` is a pickle, so
  it is executed the moment it is loaded: its sha256 is checked against the
  `<sha12>` in the URL's tag **before** the bytes are written or deserialized,
  and a mismatch aborts startup with nothing written to disk. That anchor has
  to come from the URL (deployer configuration, `render.yaml`) rather than
  from the manifest or the frozen golden, because those are fetched from the
  same store as the artifact and would be substituted along with it. A store
  whose URL carries no tag must set `FRAUD_CHAMPION_SHA256`; an unpinned
  remote fetch is refused. The corollary: **promoting and publishing a new
  champion does not change what this service serves** until `FRAUD_CHAMPION_URL`
  points at it. The old champion stays correctly pinned to its own release and
  keeps passing every check, so the staleness would otherwise be silent
  (`docs/promotion.md` → Shipping a promoted champion). Two things close that
  hop, and which one applies depends on the credentials the workflow has:
  `deploy.yml`'s *Point the host at this release's champion* step `PUT`s
  `FRAUD_CHAMPION_URL` through Render's API and then verifies both that it took
  and that no other environment variable moved; with no `RENDER_API_KEY` /
  `RENDER_SERVICE_ID` it skips, says so, and the verify step fails the release
  rather than letting it drift. For a model change with no release tag,
  `deploy-champion.yml` does the same from `configs/serving.yaml` (ADR 0012).
  Setting it by hand is the fallback when neither workflow is credentialed, not
  the normal path. The startup parity check still runs afterwards,
  but it guards *correctness* — that the artifact reproduces its frozen
  probabilities — not provenance. The URL pins the served weights by content
  hash, so what a running service serves is readable from its environment. A store that
  needs a bearer token takes it from `FRAUD_CHAMPION_TOKEN`. Publishing the
  weights is a choice: this is a portfolio model on public Kaggle data and
  every metric is already in the README, so a private store bought nothing
  but an extra account.
- `FRAUD_API_KEY` is **not** set on the demo: a dashboard that needs a key
  is not a demo. The upload cap, per-client rate limits and the 60 s time budget
  protect scoring. `FRAUD_ADMIN_API_KEY` is not
  set either, which *closes* `POST /outcomes` and `GET /audit/recent`
  (403): anonymous visitors can score but cannot write to the delayed-label
  store or read other visitors' scored rows. Setting the admin key on the
  host re-enables both for whoever holds it — and is what the scheduled
  monitoring job needs (ADR 0011): give Render `FRAUD_ADMIN_API_KEY` and
  GitHub the same value as the `FRAUD_ADMIN_API_KEY` secret, and
  `monitor.yml` starts reporting daily on the live service. Until then the
  job runs and reports that there is nothing to monitor. On 0.1 CPU uploads
  much beyond the 200-row sample run into the time budget.

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

From then on `uv run python scripts/release.py vX.Y.Z --full-checks` cuts a release
whose tag posts the deploy hook **with `?ref=<the tag's commit>`**, waits until
the live `/health` reports that same commit, runs `deploy_check.py` against the
live URL and cuts the GitHub release. The live URL goes into the README's
*Live demo* line.

A bare deploy hook builds whatever the tip of `branch: main` happens to be when
Render starts the build, which need not be the commit the tag names and CI
checked — a merge landing during the run, or a tag cut on anything but the tip,
ships untested code. Pinning the SHA closes that; the hook answers 200 only for
a valid commit, so a bad ref fails the workflow instead of deploying.

The served **version** cannot confirm which commit is live, because every commit
between one release bump and the next carries the same version. `/health`
therefore reports `build_commit` from Render's `RENDER_GIT_COMMIT`
(`FRAUD_BUILD_COMMIT` on other hosts, `null` where the host says nothing), and
the workflow fails unless it equals the commit it asked for.

## Fly.io (alternative)

```bash
brew install flyctl
flyctl auth login
flyctl apps create fraud-risk-scoring
flyctl secrets set FRAUD_API_KEY=…            # optional
flyctl volumes create fraud_audit --size 1    # persistent audit trail (mounted at /app/audit)
uv run python scripts/release.py vX.Y.Z --full-checks --fly-image
# also builds + pushes registry.fly.io/<app>:vX.Y.Z
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
scripts/release.py vX.Y.Z --full-checks   (the release machine)
  ├─ preconditions: main, clean tree, pyproject version == X.Y.Z, tag unused,
  │                 champion reproduces its golden
  ├─ fixture tests → slow real-data tests → ruff → format → mypy
  ├─ [--fly-image] flyctl deploy --build-only --push --image-label vX.Y.Z
  └─ git tag -a vX.Y.Z && git push origin vX.Y.Z
                │
                ▼  .github/workflows/deploy.yml (on the tag)
  checks   ci.yml (lint, types, tests, fixture image, container health)
  render   POST the deploy hook ?ref=<tag's commit> → Render builds exactly that commit
           from deploy/hosted/Dockerfile, the service fetches the champion at startup
           → wait until /health reports that commit → scripts/deploy_check.py
  fly      flyctl deploy --image registry.fly.io/<app>:vX.Y.Z → deploy_check.py
  release  gh release create vX.Y.Z --generate-notes
```

A **model** change does not need a release tag. `deploy-champion.yml` runs when
`configs/serving.yaml` changes on `main` (ADR 0012): it reads `champion_sha256`,
checks that `champion-<sha>` exists, points `FRAUD_CHAMPION_URL` at it, redeploys
the current commit so the service fetches it, and then requires the live artifact
to be that digest. It shares the `render-deploy` concurrency group with the
release workflow, so a release and a model change never race for the host, and it
runs in the `production` environment — add required reviewers there to hold every
model change for approval, or leave it unprotected to deploy on merge.

The weights are git-ignored and are not in any image a public runner
builds. (They are *fitted* on one when the retraining workflow runs, ADR 0012;
what protects the artifact is the digest in its release tag, which the service
pins the download against before it deserializes anything, and the golden it
must reproduce — never the runner.) On the Render leg the host builds an image without them and the
service fetches the champion from its GitHub release at startup;
on the Fly leg the runner deploys an image built where the champion is.
The release job waits for checks and both deployment legs. Checks must pass;
each configured deployment must finish successfully, including its live
verification. Failed or cancelled deployments block the GitHub release.
A leg whose variables are unset is skipped and does not block publication;
if neither host is configured, passing checks still permits a release.
For production tags, use `--full-checks` on the release machine. It adds
`uv run pytest -q -m slow` after the fixture suite and before linting or
tagging. The slow suite requires both IEEE training CSVs in `data/raw/`,
the served artifact from `configs/serving.yaml`, its frozen golden and
manifest, and the source training pipeline named by that manifest. Having
only `models/champion/` is not sufficient for all four slow tests.
Failures, skips (including missing data or artifacts), or an empty slow
suite stop the release before tag creation. No model is trained by these tests.

Without the flag, the preflight retains the fixture-only test run used by CI.
`--skip-checks` skips the test/lint/type commands, cannot be combined with
`--full-checks`, and is not a substitute for the production gate.
`scripts/release.py vX.Y.Z --full-checks --dry-run` prints the sequence
without running checks or publishing the tag; version/tag preconditions and
the existing champion verification still apply.

## Hardening for a public URL

| protection | where | behaviour |
|---|---|---|
| scoring key | `FRAUD_API_KEY` (an environment variable on the host; unset on the free demo) | when set, `/predict*` and `/explain` require a matching `X-API-Key` (constant-time compare) → 401 before the body is read; `/health`, `/model-info` and the dashboard stay open, and `/health` reports `auth: api_key`. The dashboard shows a key field when the server asks for one and keeps it in the browser only. Unset = open, for the demo and local use. |
| admin key | `FRAUD_ADMIN_API_KEY` (unset on the free demo) | `POST /outcomes` and `GET /audit/*` are operational endpoints — one writes the delayed-label store, the others read scored rows (`/audit/recent`) and build the monitoring report over them (`/audit/monitor`, ADR 0011). They answer only to this key, never to the scoring key, and while it is unset they return 403 (`/health` reports `admin: disabled`). Fail-closed: a host with no configuration exposes nothing beyond scoring. `deploy_check.py` asserts an anonymous `/audit/recent` is refused on every deployment. |
| upload cap | `serving.yaml` `max_upload_bytes` (25 MB) | declared length checked, then the stream abandoned the moment it exceeds the cap → 413; the row limit stops the parser one row past 5,000 → 422 |
| time budget | `serving.yaml` `request_timeout_s` (60 s) | a guarded call past the budget → 504. It bounds the client's wait; a scoring call already running in the thread pool finishes on its own (`/health` stays responsive, as the check below shows). |
| request IDs | `X-Request-ID` honoured or generated | echoed on every response, stored on every audit row and request record, present in every log line |
| structured logs | logger `fraud.serve.requests` | one JSON line per guarded call — `ts, request_id, method, path, status, latency_ms, rows, client` — success, 401, 422, 429 and 504 alike; Fly ships stdout to `flyctl logs` |
| metrics endpoint | none | there is no `/metrics`, no collector and no dashboard. The signals one would scrape (latency p50 / p95 / max, error rate, throughput per endpoint) are in the `requests` table and the monitoring report instead; the reasons that stack is not wired are in `docs/monitoring.md` → *Why there is no metrics collector, dashboard or tracing stack* |
| proxy concurrency (Fly only) | `fly.toml` `[http_service.concurrency]` soft 20 / hard 50 | Fly's proxy queues then refuses beyond the hard limit per machine; the benchmark (README → Performance) is why 20 is the soft limit |

Application rate limits apply on Render, Fly and local runs, through
`configs/serving.yaml` → `rate_limits`:

| POST endpoint | requests per client in any 60 seconds |
|---|---:|
| `/predict` | 60 |
| `/predict/csv` + `/predict/batch` (shared allowance) | 5 |
| `/explain` | 10 |

Authentication runs first. Every admitted attempt consumes a slot, including
invalid bodies and calls that later time out. At the limit the service returns
JSON `429` with `Retry-After` (whole seconds) and `X-Request-ID`, before reading
the body or running inference. Rejections appear in the request audit and JSON
logs with zero scored rows. Keys do not bypass quotas. Health checks, static
pages, model info and admin routes retain their existing access rules.

The limiter uses a monotonic clock and rolling windows in memory, with no new
dependency. It holds at most 10,000 client/route-class buckets, removes expired
entries, and refuses new buckets while full rather than evicting active quotas.
Limits reset on restart and are **per process**: keep one worker and one instance
on this demo. Clients sharing an IP share a quota; this is modest abuse protection,
not a distributed denial-of-service defense or a concurrency cap.

One process is a **correctness** requirement, not only a rate-limiting one, and the
deployment files say so where they start the service: the review budget is charged
against this process's SQLite audit trail, so a second worker or a second instance
would hand the same transaction a different action (`docs/model_card.md`). There is
deliberately no worker-count setting to raise — `render.yaml` used to carry a
`UVICORN_WORKERS` variable that nothing read, which suggested the opposite. Making
this service scale horizontally means sharing that state first, and sharing it
correctly means *reserving* budget transactionally rather than reading a count and
then deciding; durable storage alone removes the amnesia but not that race.

The shortlist, if that day comes, and they are not interchangeable: a Postgres
transaction (`SELECT … FOR UPDATE`) makes the reservation atomic at the cost of a
round trip on every scored batch; a Redis counter is faster but splits the count
from the record of *which* transactions were reviewed, so the two need
reconciling against the audit trail; a central decision service takes the policy
out of the scoring path entirely, which is the cleanest and the most
infrastructure. The same applies, less seriously, to the rate limiter: replicas
there degrade a protection (N replicas, N times the limit) rather than change a
decision about a customer's transaction, so it ranks below the budget.

On Render (`RENDER=true`, supplied by the platform), the client key comes from
`CF-Connecting-IP`, which Render documents as overwritten by its Cloudflare edge.
See [Render's client-IP guidance](https://render.com/articles/host-pocketbase-on-render#making-pocketbase-see-the-real-client-ip).
Missing or malformed values share a fallback bucket. The application does not
parse `X-Forwarded-For`. Outside Render it uses the ASGI connection address;
Uvicorn's trusted-proxy configuration controls any rewriting of that address.
Do not set `RENDER=true` on a directly reachable local server: callers could then
forge the trusted header. Private-network callers bypassing Render's edge must
be trusted. The request log uses the same resolved client identity as the limiter.

For a deliberate **local** load benchmark, set `rate_limits.enabled: false` in
`configs/serving.yaml` before starting the server, and restore it afterwards.
Otherwise the benchmark measures quota rejections along with scoring latency.
The earlier throughput results below predate application rate limiting.

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
