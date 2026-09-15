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
# make sure the served artifact and its frozen sample exist locally
ls models/xgb_f5_capacity_calibrated.joblib models/xgb_f5_capacity_calibrated_frozen_*.json

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

1. Train, calibrate, threshold, freeze (`README.md` → *Reproduce the model*).
2. Point `configs/serving.yaml`, `Dockerfile` and `.dockerignore` at the new
   artifact name (one `sed`, see PR #27 for the pattern).
3. `flyctl deploy --remote-only`; run `deploy_check.py`.

## Local alternative

```bash
docker build -t fraud-risk-scoring . && docker run --rm -p 8000:8000 fraud-risk-scoring
```

Identical image, identical startup check.
