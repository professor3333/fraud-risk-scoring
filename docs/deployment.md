# Deployment

The service is packaged as a Docker image (`Dockerfile`: runtime deps from
`uv.lock`, non-root user, healthcheck, the calibrated artifact and its frozen
sample). It runs anywhere Docker runs; the public deployment target is Fly.io.

## Why Fly.io, and what is and is not published

- The image is pushed to Fly's **private** registry and served from there:
  the endpoint is public, the model weights are not redistributed. (A
  Hugging Face Space or a public GHCR image would expose the artifact, which
  was trained on competition data.)
- Scale-to-zero (`auto_stop_machines`) keeps an idle demo essentially free;
  the 1 GB VM the model needs is a paid size on Fly (a few dollars a month
  while running). A card on the Fly account is required.

## One-time setup (needs the account owner)

```bash
brew install flyctl                 # installed already on the build machine
flyctl auth login                   # opens the browser; the only step that needs you
flyctl apps create fraud-risk-scoring
```

## Deploy

```bash
# make sure the champion directory exists locally (written by scripts/promote.py)
ls models/champion/

flyctl volumes create fraud_audit --size 1   # persistent audit trail (mounted at /app/audit)
flyctl deploy --remote-only          # builds the Dockerfile on Fly's builder, pushes, starts
uv run python scripts/deploy_check.py https://fraud-risk-scoring.fly.dev
```

`deploy_check.py` asserts: `/health` is `ok` with `parity_rows > 0` (the
startup parity check ran), `/model-info` reports the same version, and the
synthetic sample CSV scores 200 rows with the same model version. If the
artifact in the image does not reproduce its frozen golden, the machine
never becomes healthy and the deploy fails — which is the intended
behaviour.

## Updating the model

1. Train, calibrate, freeze (`README.md` → *Reproduce the model*).
2. `uv run python scripts/promote.py --run-name <run>` — the acceptance
   gates (`docs/promotion.md`); on success `models/champion/` is replaced
   and the registry alias moves. Nothing in `configs/`, the `Dockerfile` or
   `.dockerignore` changes.
3. Release (below), or for an unversioned deploy `flyctl deploy
   --remote-only` and `deploy_check.py`.

To revert, promote the previous champion again; it goes through the same
gates.

## Release and continuous deployment

```
scripts/release.py vX.Y.Z   (the machine that holds models/champion)
  ├─ preconditions: main, clean tree, pyproject version == X.Y.Z, tag unused,
  │                 champion reproduces its golden
  ├─ tests → ruff → format → mypy                      (§14 ship sequence)
  ├─ flyctl deploy --build-only --push --image-label vX.Y.Z
  │     → registry.fly.io/fraud-risk-scoring:vX.Y.Z   (private; weights never leave it)
  └─ git tag -a vX.Y.Z && git push origin vX.Y.Z
                │
                ▼  .github/workflows/deploy.yml (on the tag)
  ci.yml (lint, types, tests, fixture image, container health)
  flyctl deploy --image registry.fly.io/<app>:vX.Y.Z    needs secrets.FLY_API_TOKEN, vars.FLY_APP
  scripts/deploy_check.py https://<app>.fly.dev         needs secrets.FRAUD_API_KEY when the service is keyed
  gh release create vX.Y.Z --generate-notes
```

Why two halves: the model weights are git-ignored and must not reach a
public runner or a public image, so the runner deploys an image it cannot
build. The build happens where the champion is, gated by the same tests
CI runs; the runner only ever moves a labelled image and checks the
result. Until `FLY_API_TOKEN` and `FLY_APP` are set (`flyctl tokens create
deploy`; repository settings → secrets and variables), a tag still runs
the checks and cuts the GitHub release, and the deploy job is skipped.
`scripts/release.py --dry-run` prints the sequence without doing it;
`--skip-image` tags an image pushed earlier.

## Hardening for a public URL

| protection | where | behaviour |
|---|---|---|
| API key | `FRAUD_API_KEY` (a Fly secret: `flyctl secrets set FRAUD_API_KEY=…`) | when set, `/predict*`, `/explain`, `/outcomes` and `/audit/*` require a matching `X-API-Key` (constant-time compare) → 401 before the body is read; `/health`, `/model-info` and the dashboard stay open, and `/health` reports `auth: api_key`. The dashboard shows a key field when the server asks for one and keeps it in the browser only. Unset = open, for local use. |
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
