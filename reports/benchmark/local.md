# API benchmark — local

Target local uvicorn (one worker); 2026-09-16 10:03; Darwin arm64, 8 CPUs.

| scenario | p50 | p95 / max |
|---|---:|---:|
| cold start (launch → healthy /health) | 2,125 ms | — |
| single /predict, sequential | 51 ms | 81 ms |
| /predict/batch, 1,000 rows | 467 ms | 543 ms |
| /predict/csv, 200 rows | 565 ms | 591 ms |
| /predict/csv, 1,000 rows | 1,427 ms | 1,473 ms |
| /predict/csv, 5,000 rows | 5,838 ms | 5,861 ms |
| 20 concurrent clients, single /predict | 1,223 ms | 1,383 ms |
| 20 concurrent 200-row CSV uploads | 7,555 ms | 11,178 ms |

- throughput, 20 concurrent single predictions: **17 req/s** (0 errors of 200)
- concurrent CSV: 358 rows/s
- 5,000-row CSV body: 4.7 MB
- peak resident memory of the server process: **477 MB**
