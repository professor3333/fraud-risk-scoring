# API benchmark — docker_1cpu_1gb

Target http://127.0.0.1:8012; 2026-09-16 10:05; Darwin arm64, 8 CPUs.

| scenario | p50 | p95 / max |
|---|---:|---:|
| container cold start (docker run → healthy /health, measured outside) | 3,466 ms | — |
| single /predict, sequential | 46 ms | 77 ms |
| /predict/batch, 1,000 rows | 594 ms | 623 ms |
| /predict/csv, 200 rows | 513 ms | 536 ms |
| /predict/csv, 1,000 rows | 1,378 ms | 1,460 ms |
| /predict/csv, 5,000 rows | 6,428 ms | 6,575 ms |
| 20 concurrent clients, single /predict | 933 ms | 1,188 ms |
| 20 concurrent 200-row CSV uploads | 6,624 ms | 10,494 ms |

- throughput, 20 concurrent single predictions: **21 req/s** (0 errors of 200)
- concurrent CSV: 381 rows/s
- 5,000-row CSV body: 4.7 MB
- peak container memory (docker stats, sampled every second): **538 MiB** under --memory 1g --cpus 1
