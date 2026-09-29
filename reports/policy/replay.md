Champion xgb_f5_capacity+sigmoid (7af85ec92813), validation days 123-152, 85,044 transactions, block >= 0.42, review budget 200/day. Daily and hourly replays agree on 95.7% of actions.

| policy | review_catch_rate | reviewed_per_day | review_precision | recall_block_plus_review | fraud_amount_caught_share | cost_per_transaction |
|---|---|---|---|---|---|---|
| fixed band (old promotion) | 1.0000 | 200.0000 | 0.1480 | 0.7760 | 0.7740 | 1.8478 |
| fixed band (old promotion) | 0.9000 | 200.0000 | 0.1480 | 0.7452 | 0.7338 | 2.0805 |
| fixed band (old promotion) | 0.7000 | 200.0000 | 0.1480 | 0.6836 | 0.6536 | 2.5458 |
| pooled window (old reference) | 1.0000 | 200.0000 | 0.1480 | 0.7760 | 0.7740 | 1.8478 |
| pooled window (old reference) | 0.9000 | 200.0000 | 0.1480 | 0.7452 | 0.7338 | 2.0805 |
| pooled window (old reference) | 0.7000 | 200.0000 | 0.1480 | 0.6836 | 0.6536 | 2.5458 |
| daily upload (served; gated) | 1.0000 | 200.0000 | 0.1493 | 0.7788 | 0.7765 | 1.8321 |
| daily upload (served; gated) | 0.9000 | 200.0000 | 0.1493 | 0.7477 | 0.7362 | 2.0663 |
| daily upload (served; gated) | 0.7000 | 200.0000 | 0.1493 | 0.6856 | 0.6554 | 2.5347 |
| hourly requests (served) | 1.0000 | 200.0000 | 0.1342 | 0.7472 | 0.7414 | 2.0223 |
| hourly requests (served) | 0.9000 | 200.0000 | 0.1342 | 0.7193 | 0.7045 | 2.2359 |
| hourly requests (served) | 0.7000 | 200.0000 | 0.1342 | 0.6635 | 0.6308 | 2.6631 |
| 15-minute requests (served) | 1.0000 | 200.0000 | 0.1195 | 0.7167 | 0.6854 | 2.3401 |
| 15-minute requests (served) | 0.9000 | 200.0000 | 0.1195 | 0.6919 | 0.6541 | 2.5219 |
| 15-minute requests (served) | 0.7000 | 200.0000 | 0.1195 | 0.6421 | 0.5916 | 2.8854 |
