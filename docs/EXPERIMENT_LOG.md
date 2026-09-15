# Experiment log — hypothesis · experiment · result · decision

The compact ledger. Long-form hypotheses, configs and interpretation are in
`docs/experiments.md`; numbers and artifacts are in MLflow
(`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`). All results are
validation PR-AUC on the frozen temporal split unless stated; "paired" means
the seed-paired mean over seeds 42 / 1 / 2 (ADR 0003).

| # | Hypothesis | Experiment | Result | Decision |
|---|---|---|---|---|
| E001 | The no-skill floor equals the positive rate | `DummyClassifier(prior)` | PR-AUC 0.034, ROC 0.50 | Reference |
| E002b | A handful of interpretable columns carry linear signal | logistic regression, 26 columns | 0.288, no train/val gap | Reference |
| E002 | All raw columns help a linear model | logistic regression, 423 columns | 0.402, 16 min | Reference |
| E003 | Fraud is interaction-heavy; trees beat linear on the same columns | XGBoost, defaults, raw columns | **0.570** (+0.17) | Keep — first XGBoost |
| E004 | Seed noise is small relative to model steps | E003 at 4 seeds | sd 0.002, range 0.005 | Acceptance rule set: ≥ 0.01, or paired ≥ 0.005 |
| E005 | Hour of day adds signal (10.6 % fraud at "hour 7") | + `hour`, `weekday` | +0.006, paired +0.002 | Reject — `D8`/`D9`/`V` already carry it |
| E006 | "How common is this card / device / domain" separates rare-entity fraud | frequency tables fit on train, 12 cols | paired **+0.009**, ROC +0.007 | Keep |
| E007 | Card-history velocity (strictly earlier rows) adds beyond provider counts | `(card1, addr1, day−D1)` entity, 5 features | paired −0.002 | Reject — provider `C`/`D` already summarise it |
| E008 | Defaults are not a tuned point; deeper trees fit the interactions | 16-trial random search, expanding-window CV | CV +0.019 → val **0.616** (+0.038) | Keep — depth 8, 800 trees |
| E009 | `V` dominates gain but is redundant; `C`/`card` irreplaceable | gain + permutation + group ablation | `V`: 76 % gain, −0.005 removed; `card` −0.053, `C` −0.049 | Informs; no change |
| E010 | A random split over-reports because it shares accounts and weeks | stratified random vs temporal, same model | 0.810 vs 0.616 | Control only — never for selection |
| E011 | Trees need no imputation | median impute + indicators before XGBoost | −0.003, 2× slower | Reject |
| E012 | Rare one-hot levels are harmless; grouping is a safer contract | `min_frequency=100` | paired +0.002 | Neutral — recommended at next retrain |
| E013 | Missing-field counts summarise informative missingness | + 3 count columns | −0.001 | Reject |
| E014 | Amount structure (cents, round, integer) carries signal | + 5 amount columns | −0.001 | Reject |
| E015 | Purchaser = recipient domain and provider families help | + e-mail flags & families | −0.006 | Reject |
| E016 | Card × address / e-mail is closer to "an account" than `card1` | 4 composite keys, frequency-encoded | paired **+0.006**, all seeds + | Keep — shipped (v0.2.0) |
| E017 | A coarser entity with std / max finds what E007 missed | `card1+addr1` history, 8 features | −0.001 | Reject — question closed |
| E018 | Accumulating sets tracks the per-set verdicts; Vesta's features carry the model | cumulative ladder + family cuts | F5/F6 only real steps; `C/D/M/V` worth 0.19, mine 0.02–0.03 | Informs |
| E019 | Cumulative F0–F6 (0.626) is a new best | seed pairs vs E016 | paired +0.000 | Reject — lucky seed |
| E020 | `V` is optional once entity frequency is modelled | shipped set without `V` (100 inputs) | paired −0.003 | Neutral — compact option, not shipped |
| E021 | One-axis sweeps show underfit → capacity → overfit → regularise | depth / η / mcw / subsample / colsample / L1 / L2 / trees | underfit side as expected; validation never turns over (depth 12 0.629, train 1.000 model 0.627); regularisers only hurt | Gap = account memorisation, not a metric; direction → E022 (`docs/xgboost_progression.md`) |
| E022 | More capacity (depth 12, mcw 1, 1600 trees) beats the shipped model | seeds 42 / 1 / 2 vs E016 | paired **+0.017**, holds all validation month; test +0.004, operating points slightly worse | Keep — shipped (v0.3.0); drift-aware validation protocol is the open follow-up |
| E023 | The adjacent window hides multi-month decay; horizons change the winner | rolling backtests (ADR 0008): 3 cut-offs × gaps 0–60 d, seeds 42/1/2 | E022 wins all 10 windows; lead +0.013 adjacent → +0.006 at 60 d; robust paired +0.007 | Keep E022 — confirmed on validation-only, drift-aware evidence |
| E024 | A month-stale model loses materially; a challenger cycle promotes monthly | offline lifecycle: retrain → calibrate → threshold → challenge → promote → freeze | incumbent 0.522 vs challenger 0.647 on the next month (+0.125); block threshold 0.47 → 0.425 | Monthly retraining is the largest lever in the project; lifecycle is code (`scripts/retrain.py`) |
| — | Isotonic calibration fixes the tail | isotonic vs sigmoid on OOF scores | isotonic ties cost 0.011 PR-AUC; sigmoid Brier 0.0195→0.0191 | Sigmoid (ADR 0007) |
| — | A review action beats decline-only; ranks hold under drift, thresholds don't | three-band policy sized to a budget, checked on test | 200 reviews/day: recall 0.77, cost −30 % vs decline-only; test block precision 0.80→0.71, review volume +25 % | Block by threshold, review by daily rank (`docs/review_policy.md`) — implemented in `/predict/batch` and `/predict/csv` |
| — | The cost-optimal threshold is far below 0.5 | amount-weighted cost curve | 0.08 (flat 0.06–0.145); −52 % cost vs approve-all | 0.08 (ADR 0006) |
| — | Errors are informative: what fools the model? | final bundle + inspection of HC FP / FN / TP on test | FP = the fraud archetype done by real customers; FN = propagated labels (68 % share an entity with other fraud) + established-card takeover | Review the archetype, don't block; per-entity deviation on the `W` slice is the next experiment (`docs/error_analysis.md`) |
| — | The model transfers one month further out | reporting-window evaluations of E008, E016, E022 (ADR 0002 log: 7 consultations, no selection) | E008 0.553; E016 0.557; E022 0.561 (validation gain +0.017 → +0.004) | Reported with the not-blind caveat; retrain monthly; validation horizon is the methodology gap |
| — | Production must give the same probability as training | frozen 50-row golden, checked at API startup and in tests | max diff 0.0; tampered golden or swapped artifact refuses to start | Mandatory (`fraud.serve.parity`) |
