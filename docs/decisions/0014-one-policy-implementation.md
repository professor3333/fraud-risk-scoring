# ADR 0014 — One policy implementation, and the product is a daily ranking tool

**Date:** 2026-09-29 · **Status:** accepted · **Amends:** ADR 0010 (what promotion
prices), ADR 0011 (what the monitoring reference expects)

## Context

Three pieces of code each said what "the policy" does, and they disagreed:

| where | what it computed |
|---|---|
| serving (`/predict/batch`, `/predict/csv`) | block ≥ threshold; review the top remaining scores **per transaction day**, capacity released through the day and charged across requests (ADR 0013) |
| promotion gates (`candidate_metrics`) | a **fixed review threshold** sized so the validation window averaged the budget per day |
| monitoring reference (`build_reference`) | the top `budget × days` of the **whole window pooled**, so a busy day could take a quiet day's reviews |

Promotion's cost and recall gates therefore priced a policy the service
never ran. They were also blind to the one thing the served policy depends
on that the others do not: **how requests arrive**. A request is ranked
only against itself and what its day has released so far, so a low score
sent in the morning can hold a slot a higher evening score needed.

## Decision

1. **One implementation.** `fraud.evaluate.policy.replay_served_policy`
   replays the served rank policy over any rows: the release rule
   (`released_capacity`) and the per-request ranking
   (`apply_daily_rank_policy`) are the same functions serving calls, and a
   test sends the same rows to the service as the same requests and asserts
   identical actions, whole-day and hourly. Promotion and the monitoring
   reference both use it. `policy_outcome` prices any set of actions.
2. **The product is a daily ranking tool.** Its primary use is the
   dashboard: upload a day's transactions, get the day's queue. Promotion
   gates on the replay with **each day as one request**, which is exactly
   "block by threshold, review the day's top N".
3. **Online use is supported and measured, not gated.** The same replay with
   hourly requests is reported as `stream_recall_block_plus_review` and
   `stream_cost_per_transaction` beside the gated numbers.
4. **Ranking and policy metrics are kept apart.** PR-AUC, ROC-AUC,
   recall/precision at k per day and calibration describe scores. Block
   precision, recall with review, amount caught and cost describe the
   served policy's outcome.
5. **The review catch rate is a stated assumption.** `policy_outcome` takes
   the share of reviewed fraud an analyst actually stops (default 1.0, the
   cost model's original assumption). The replay report prices 1.0, 0.9 and 0.7.
6. **Champions are re-measured, not recalled.** `scripts/promote.py` now
   measures the incumbent on the same rows with the same code, as the
   retraining cycle already did, instead of comparing against numbers its
   manifest stored under an older definition.

## Evidence

`scripts/replay_policy.py` → `reports/policy/replay.md`, champion
`7af85ec92813`, validation days 123–152, block ≥ 0.42, 200 reviews/day,
catch rate 1.0:

| policy | review precision | recall (block + review) | fraud amount caught | cost / transaction |
|---|---|---|---|---|
| fixed band (old promotion) | 0.148 | 0.776 | 0.774 | 1.848 |
| pooled window (old reference) | 0.148 | 0.776 | 0.774 | 1.848 |
| **daily upload (served, gated)** | **0.149** | **0.779** | **0.777** | **1.832** |
| hourly requests (served) | 0.134 | 0.747 | 0.741 | 2.022 |
| 15-minute requests (served) | 0.120 | 0.717 | 0.685 | 2.340 |

At a catch rate of 0.7 the daily policy's recall falls to 0.686 and its
cost rises to 2.535 (+38 %).

## Consequences

- The old gates happened to be a close proxy for daily uploads (within
  0.003 recall and 0.016 cost), so no past promotion decision is overturned.
  They were not a proxy for anything else.
- Used as an online queue, the same model and budget catch 3.2 points less
  fraud at hourly granularity and 6.2 points less at 15 minutes. That is a
  property of the policy, not the model. If the product ever becomes an
  online queue, the fix belongs in the policy (for example, holding capacity
  for later hours, or re-ranking a day's open queue), and it must be gated
  on the stream replay, not the daily one.
- The monitoring reference's action shares now come from the served policy
  per day. Live shares still depend on how the day's traffic was chunked,
  which the report cannot see, so an action-share drift flag on chunked
  traffic can be arrival order rather than the model.
- The cost model's "reviewed fraud is caught" assumption is now visible in
  the numbers it produces. An operator with a measured catch rate should
  set it, not argue with the 1.0.
