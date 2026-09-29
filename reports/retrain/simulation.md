# Simulated retraining lifecycle

Snapshot `data/raw`; clock days 92, 110, 122, 183; labels final on the day they occur (`--label-maturity-days 0`); workspace `<a fresh temp dir>`. Nothing outside the workspace was read or written except the data cache.

| day | trigger | ran | outcome | published | service started from it | seconds |
|---|---|---|---|---|---|---|
| 92 | `bootstrap` | yes | PROMOTED | champion-611199307f76 | xgb_f5_capacity_through_day62+sigmoid@611199307f76 · bands 0.0853/0.5100 | 429 |
| 110 | `too-little-new-data` | no | waited | — | — | 3 |
| 122 | `calendar` | yes | PROMOTED | champion-ece2ed287853 | xgb_f5_capacity_through_day92+sigmoid@ece2ed287853 · bands 0.0837/0.4800 | 581 |
| 183 | `reporting-window` | no | waited | — | — | 3 |

## Why each step did what it did

- **day 92** — bootstrap: no champion to beat
- **day 110** — only 18 new training days since the champion's cut-off (day 62); the step is 30
- **day 122** — challenger PR-AUC +0.0754 vs champion on days 93–122 (clears the 0.005 margin)
- **day 183** — the evaluation month 154–183 reaches into the reporting window (day 153+). Consuming it is a consultation of the held-out window (ADR 0002): set cycle.allow_reporting_window in a reviewed commit and record it in the log

## Policy change proposed on day 92

```diff
--- configs/serving.yaml
+++ configs/serving.yaml
@@ -5,12 +5,12 @@
 # model_version, model_info and bands below are replaced by the manifest's values —
 # they remain here as the fallback for a directory without one (CI's fixture artifact).
 model_path: models/champion/model.joblib
-model_version: xgb_f5_capacity+sigmoid
+model_version: xgb_f5_capacity_through_day62+sigmoid
 # The artifact these bands were measured on. Checked when the champion is fetched from
 # FRAUD_CHAMPION_URL (ADR 0012): the host chooses which artifact to load, the bands come
 # from this reviewed file, and the service refuses to serve a pairing nobody approved.
 # scripts/retrain_cycle.py writes this line and the bands together.
-champion_sha256: 7af85ec92813
+champion_sha256: 611199307f76
 # Startup parity check against the frozen sample next to the artifact (G8).
 require_parity: true
 # Production policy (docs/review_policy.md): block at bands.block (the 80 % precision
@@ -23,8 +23,8 @@
 # digest-pinned, and the service refuses to take its bands (docs/security.md). They
 # must equal the promoted champion's manifest bands exactly or startup fails.
 bands:
-  review: 0.06224176041919635
-  block: 0.42
+  review: 0.08528224108920522
+  block: 0.51
 # Facts about the served model, for /model-info (numbers from reports/ and MLflow).
 model_info:
   model: xgboost
```

## Policy change proposed on day 122

```diff
--- configs/serving.yaml
+++ configs/serving.yaml
@@ -5,12 +5,12 @@
 # model_version, model_info and bands below are replaced by the manifest's values —
 # they remain here as the fallback for a directory without one (CI's fixture artifact).
 model_path: models/champion/model.joblib
-model_version: xgb_f5_capacity_through_day62+sigmoid
+model_version: xgb_f5_capacity_through_day92+sigmoid
 # The artifact these bands were measured on. Checked when the champion is fetched from
 # FRAUD_CHAMPION_URL (ADR 0012): the host chooses which artifact to load, the bands come
 # from this reviewed file, and the service refuses to serve a pairing nobody approved.
 # scripts/retrain_cycle.py writes this line and the bands together.
-champion_sha256: 611199307f76
+champion_sha256: ece2ed287853
 # Startup parity check against the frozen sample next to the artifact (G8).
 require_parity: true
 # Production policy (docs/review_policy.md): block at bands.block (the 80 % precision
@@ -23,8 +23,8 @@
 # digest-pinned, and the service refuses to take its bands (docs/security.md). They
 # must equal the promoted champion's manifest bands exactly or startup fails.
 bands:
-  review: 0.08528224108920522
-  block: 0.51
+  review: 0.08366372712198406
+  block: 0.48
 # Facts about the served model, for /model-info (numbers from reports/ and MLflow).
 model_info:
   model: xgboost
```
