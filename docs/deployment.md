# Deployment

The service is packaged as a Docker image (`Dockerfile`: runtime deps from
`uv.lock`, non-root user, healthcheck, the champion directory and its frozen
sample). It runs anywhere Docker runs. Two public targets are wired:

| | Hugging Face Space (default) | Fly.io (alternative) |
|---|---|---|
| cost | **free** — CPU basic (2 vCPU, 16 GB), no card | 1 GB machine is paid; a card is required |
| the weights | fetched at startup from a **private** HF model repo with a token; the public Space repo holds three files and no model | baked into an image in Fly's **private** registry |
| idle behaviour | sleeps after inactivity; first request wakes it (build is cached; container start ≈ 3 s + a 41 MB fetch) | scale-to-zero |
| persistence | none on the free tier: the audit trail is ephemeral | a volume for the audit trail |
| CD leg (`deploy.yml`) | `space` job: needs `vars.HF_SPACE`, `secrets.HF_TOKEN` | `fly` job: needs `vars.FLY_APP`, `secrets.FLY_API_TOKEN` |

Fly was the original target; its config (`fly.toml`) stays for anyone
with an account. The zero-payment requirement is what moved the default.

## The Space

```
GitHub repo ──(git clone at FRAUD_REF)──► Space build ──► image without weights
                                                              │ startup
private HF model repo (models/champion/) ──(HF_TOKEN)──► fetch_champion() ──► parity check ──► serve
```

- `deploy/space/` is the whole Space repository: a `Dockerfile` that clones
  this repository at `FRAUD_REF` and installs it from the lock file, and a
  `README.md` with the Space metadata (`sdk: docker`, `app_port: 7860`).
- `FRAUD_CHAMPION_URL` (Space variable) = `https://huggingface.co/<user>/<model-repo>/resolve/main`;
  `HF_TOKEN` (Space secret, read scope) authorises it. `fraud.serve.app.
  fetch_champion` downloads `model.joblib`, the frozen golden, the manifest
  and the monitoring reference into `models/champion/` when the directory
  is empty, and the startup parity check runs as everywhere else — a Space
  with the wrong token or a tampered file does not come up.
- `FRAUD_API_KEY` is **not** set on the demo Space: the dashboard has to
  score without a key to be a demo. The upload cap, the time budget and
  the Space's own sleep are the protections; the audit trail is
  ephemeral, so `/outcomes` writes nothing that outlives a restart.

Verified locally end to end before any account existed: the Space image
built from GitHub, a token-checking store serving `models/champion/`,
`FRAUD_CHAMPION_URL` pointing at it → healthy in 3 s, `parity_rows` 50,
`deploy_check.py` passed, dashboard served.

## One-time setup (needs the account owner, once)

```bash
uvx --from huggingface_hub hf auth login                          # a free HF account, no card
uv run python scripts/publish_champion.py --repo <user>/fraud-risk-scoring-model
#   → creates the PRIVATE model repo, uploads models/champion/, prints FRAUD_CHAMPION_URL

# the Space: create it once (Docker SDK, public, CPU basic), then set on it
#   variable FRAUD_CHAMPION_URL = the URL printed above
#   secret   HF_TOKEN           = a read token for the model repo
uvx --from huggingface_hub hf repo create <user>/fraud-risk-scoring --repo-type space --space-sdk docker
uvx --from huggingface_hub hf upload spaces/<user>/fraud-risk-scoring deploy/space . --repo-type space

# continuous deployment from GitHub on every tag
gh variable set HF_SPACE --body <user>/fraud-risk-scoring
gh secret set HF_TOKEN --body <a write token for the Space>
```

Then `uv run python scripts/release.py vX.Y.Z` cuts a release whose tag
pushes the Space repository pinned to that tag, waits for the build, runs
`deploy_check.py https://<user>-fraud-risk-scoring.hf.space`, and cuts the
GitHub release. The live URL goes into the README's *Live demo* line.

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
   so the Space can fetch it, then release (below).

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
  space    push deploy/space/ pinned to vX.Y.Z → Space builds from GitHub, fetches the champion
           → scripts/deploy_check.py https://<user>-<space>.hf.space
  fly      flyctl deploy --image registry.fly.io/<app>:vX.Y.Z → deploy_check.py
  release  gh release create vX.Y.Z --generate-notes
```

The weights are git-ignored and must not reach a public runner or a
public image. On the Space leg the runner pushes three files and the
Space itself clones the tag and fetches the champion from the private
repo; on the Fly leg the runner deploys an image built where the champion
is. A leg whose variables are unset is skipped; the release is still cut.
`scripts/release.py --dry-run` prints the sequence without doing it.

## Hardening for a public URL

| protection | where | behaviour |
|---|---|---|
| API key | `FRAUD_API_KEY` (a Space secret or a Fly secret; unset on the demo Space) | when set, `/predict*`, `/explain`, `/outcomes` and `/audit/*` require a matching `X-API-Key` (constant-time compare) → 401 before the body is read; `/health`, `/model-info` and the dashboard stay open, and `/health` reports `auth: api_key`. The dashboard shows a key field when the server asks for one and keeps it in the browser only. Unset = open, for local use. |
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
