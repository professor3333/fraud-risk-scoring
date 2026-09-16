---
title: Fraud Risk Scoring
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Calibrated XGBoost fraud scoring with a review policy
---

# Fraud risk scoring — live demo

The inference service from
[professor3333/fraud-risk-scoring](https://github.com/professor3333/fraud-risk-scoring):
a calibrated XGBoost model on the IEEE-CIS Fraud Detection data, served by
FastAPI with a three-action review policy (approve / review / block), a
startup parity check against a frozen golden, per-prediction explanations
and a prediction audit trail.

- `/` — the analyst dashboard: upload a CSV (or load the synthetic sample),
  set the daily review capacity, inspect any row and see why it scored as
  it did.
- `/docs` — the OpenAPI schema; `/health`, `/model-info`.

The model weights are not in this Space: the service fetches them at
startup from a private repository and refuses to start unless they
reproduce their frozen golden. The Space sleeps when idle; the first
request after a sleep takes a few seconds. The audit trail here is
ephemeral (free tier, no persistent storage).
