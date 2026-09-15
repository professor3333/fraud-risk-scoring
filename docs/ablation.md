# Ablation and importance — `xgb_v2_tuned` (E008)

Evidence: `reports/ablation/`, produced by
`uv run python scripts/ablation.py --model-config configs/model/xgboost_v2_tuned.yaml`.
All numbers are validation PR-AUC (base **0.6155**). Importance is reported
from two methods that answer different questions, and they disagree in an
instructive way.

## Method 1 — gain (what the fitted model uses)

Share of total split gain by source-column group:

| group | gain share |
|---|---:|
| `V` (339 cols) | 0.763 |
| `C` (14) | 0.065 |
| identity (37) | 0.056 |
| frequency (12) | 0.041 |
| `M` (9) | 0.030 |
| `card` (6) | 0.016 |
| `D` (15) | 0.014 |
| `ProductCD` | 0.009 |
| `addr`, `dist`, amount | ≤ 0.003 |
| `has_identity` | 0.000 |

Top single features: `V258` (11.8 %), `V246` (8.1 %), `V257` (3.3 %),
`freq_addr2` (2.9 %), `V225`, `V156`, `V324`, `V147`, `V294`, `V295`, `C8`,
`C4`, `C7`, `id_28 = <missing>`.

## Method 2 — group permutation (what breaks the fitted model)

Drop in validation PR-AUC when a whole group is shuffled jointly on the
validation window (3 repeats, no refit):

| group | drop | | group | drop |
|---|---:|---|---|---:|
| `C` | **0.258** | | `M` | 0.028 |
| `V` | 0.192 | | email | 0.015 |
| `card` | 0.113 | | `dist` | 0.009 |
| `D` | 0.057 | | `ProductCD` | 0.004 |
| amount | 0.035 | | `has_identity` | 0.000 |
| `addr` | 0.033 | | | |
| identity | 0.029 | | | |

## Method 3 — ablation (what the data can do without a group)

Refit the same config with one group removed (each refit logged to MLflow
experiment `fraud-ablation`):

| removed | columns | val PR-AUC | Δ vs full | train/val gap |
|---|---:|---:|---:|---:|
| `card` | 10 | 0.5625 | **−0.053** | 0.316 |
| `C` | 14 | 0.5663 | **−0.049** | 0.331 |
| frequency | 12 | 0.6014 | −0.014 | 0.296 |
| `M` | 9 | 0.6017 | −0.014 | 0.310 |
| `addr` | 4 | 0.6047 | −0.011 | 0.304 |
| identity | 37 | 0.6069 | −0.009 | 0.300 |
| amount | 1 | 0.6070 | −0.009 | 0.300 |
| `V` | 339 | 0.6102 | −0.005 | 0.315 |
| email | 2 | 0.6129 | −0.003 | 0.301 |
| `ProductCD` | 1 | 0.6138 | −0.002 | 0.304 |
| `D` | 15 | 0.6143 | −0.001 | 0.294 |
| `dist` | 2 | 0.6200 | +0.005 | 0.298 |
| `has_identity` | 1 | 0.6201 | +0.005 | 0.298 |

(“columns” counts raw plus frequency-encoded columns removed together.)

## Where the methods disagree, and why

1. **`V` is the model's favourite and the data's most replaceable group.**
   Gain 76 %, permutation −0.19, ablation **−0.005**. Gain measures where
   the trees chose to split; permutation measures what happens when a
   fitted model's inputs are corrupted; neither asks whether the same
   information exists elsewhere. Refitting without `V` shows it does — the
   `C`, `card`, `M` and frequency columns carry nearly all of it. The
   provider's `V` features are, by their own description, engineered
   from the other columns, so this is what one would expect. Consequence:
   a **131-column model without `V`** is within 0.005 of the full model,
   which is worth a logged experiment if serving cost or explainability
   matter more than half a point of PR-AUC. Not done in this build to
   keep the ledger at one change per experiment.
2. **`C` and `card` are irreplaceable.** They rank high on permutation
   *and* ablation. `C*` are the provider's counts of entities linked to the
   card; `card1–6` identify the card itself. This is also why E007's
   entity-history features added nothing: the provider had already
   summarised the card's past into these columns.
3. **Frequency encoding earns its place a second way.** Ablation −0.014 is
   larger than the E006 seed-paired gain (+0.009), consistent with the
   tuned, deeper model making more use of it.
4. **`has_identity` is worth exactly nothing** on every method, as the EDA
   predicted (`W` never has identity; every other product almost always
   does, so `ProductCD` already says it). It stays in the contract because
   the serving path needs to know whether an identity record was supplied;
   the model just doesn't use it.
5. **`dist` and `has_identity` removal shows +0.005** — below the noise
   floor's decision band (seed sd 0.002; the rule needs ≥ 0.005 seed-paired
   to act). Recorded, not acted on.
6. **The gap does not move.** Every ablated model sits at a 0.29 – 0.33
   train/validation gap; no single group is responsible for the overfit —
   it is the depth-8 trees on 420k rows, and the learning curve shows it is
   benign (validation still rising at 800 trees).
