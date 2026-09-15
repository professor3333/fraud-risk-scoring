# Defending the decisions

The owner's study sheet. Every major decision as the question a reviewer
would ask, the answer, the evidence to point at, and the strongest objection
— because a decision you can only defend against weak objections is not
defended. Pointers are to files in this repository.

---

### "What exactly are you predicting?"

`isFraud` as the provider defines it: the transaction is, or belongs to an
account that becomes, reported as fraud — scored per transaction at
authorization, from the row and strictly earlier information (ADR 0001).

**Evidence.** 81 % of fraud rows sit in entities with ≥ 2 fraud rows; 3,118
entities are entirely fraud (`docs/eda.md` §1, §6). The label propagates.

**Strongest objection.** "Then a quarter of your false negatives are not
errors." Correct — `docs/error_analysis.md` shows 71 % of confident false
negatives share an entity with other labelled fraud while that entity is
only 17 % fraud. Under this label they are not recoverable at
authorization; the alternative (predict only the first fraud per entity)
throws away 80 % of positives and needs an entity reconstruction that is
itself a leakage question. I chose the label the business acts on.

### "Why a temporal split and why those boundaries?"

Train days 1–122, validation 123–152, test 153–183 (ADR 0002). Validation
and test are 30-day windows with ~2.9k positives each — enough for a stable
PR-AUC; both sit after the seasonal first month.

**Evidence.** E010: the same model on a random stratified split reports
0.810 vs 0.616 on the temporal split — it shares accounts (label
propagation) and weeks (drift) between train and validation. The test
month is 0.06–0.08 below validation for every candidate; that drift only
exists if you look forward in time.

**Strongest objection.** "Your validation window is adjacent to training;
it measures one-month transfer and your deployment horizon may be longer."
True, and it bit: E022's +0.017 validation gain became +0.004 on the test
month. This is recorded as the first open follow-up — a drift-aware
validation horizon, decided by ADR *before* any further selection — rather
than patched after seeing the test number.

### "Is your test set really a one-look holdout?"

No, and the documentation now says so. No model was selected on it, but
three candidates (E008, E016, E022) and two policy checks were reported on
the same later window during development — five distinct looks, logged in
ADR 0002. It is a *final temporal reporting window*: one month further out
than validation, with a small optimistic bias from having been seen. The
clean fix is to adopt a later-horizon validation protocol and re-freeze a
reporting window for a genuinely single final look; that is the first
follow-up, not something to claim retroactively.

### "No gap between train and validation?"

Features use only the row and strictly earlier rows, so a row at the
boundary is scored with exactly the history a live system would have; a gap
does not make features more valid. What a gap would model is label
immaturity — reports arriving after the training cut-off — which makes
training labels slightly better than production's by a time-invariant
amount and does not change model comparisons. Recorded as a limitation in
the model card instead of paid for with 30 days of data.

### "Why PR-AUC and not ROC-AUC, which Kaggle uses?"

3.5 % positives: ROC-AUC is dominated by how the 96.5 % negatives are
ordered among themselves and barely moves when tail precision changes. The
PR curve *is* the threshold trade-off the business acts on (ADR 0003).
ROC-AUC is always reported alongside.

**Evidence.** E006: PR-AUC +0.009 paired, ROC +0.007 — they usually agree;
E022: ROC-AUC went *down* on test (0.910 → 0.907) while PR-AUC went up —
they measure different regions.

### "How do you handle class imbalance?"

I don't re-weight or resample (ADR 0003). PR-AUC and ROC-AUC are ranking
metrics; re-weighting changes the score scale and the threshold, not the
ranking a tree learns from the same splits, and it distorts probabilities
that calibration would then have to undo. Weighting stays available as an
experiment; none was needed.

### "How do you decide a change is real?"

A validation PR-AUC gain of ≥ 0.01, or a seed-paired mean ≥ 0.005 with
every pair positive (ADR 0003, amended once — and re-applied to every
earlier decision, changing none).

**Evidence.** E004 measured a seed sd of 0.002; E012 confirmed it on the
tuned model. E019 (+0.0065 at one seed → +0.0003 paired) is the case the
rule exists for.

**Strongest objection.** "You amended the rule after seeing E016's number."
Yes: the seed-42 delta was +0.0035, below the old re-run trigger, and I
re-ran anyway. The amendment made the rule *stricter* on the outcome (all
pairs positive), and the retroactive check is in the ADR.

### "Is `has_identity` a feature? Isn't identity presence a strong signal?"

It is in the contract and worth exactly zero: 0.000 on gain, permutation and
ablation (`docs/ablation.md`). Identity exists for 0 % of product `W` rows
and ≥ 91 % of every other product, so `ProductCD` already says it. The EDA
hypothesis was right about a correlation and wrong about a conditional gain.

### "Why frequency encoding and not target encoding?"

Frequency tables fit on the training window add a notion the raw
identifiers lack ("how common is this card / device / domain") with no
label involved (ADR 0005, E006 +0.009 paired). Whole-dataset counts — the
Kaggle version — include validation and test rows and, in production, the
future. Target encoding is deferred: it must be out-of-fold *and*
time-ordered inside training or it memorises the label, and E006 left
little to gain.

### "Kaggle solutions get most of their lift from a card 'uid'. Where is yours?"

Built, tested and rejected twice (ADR 0004; E007 −0.002, E017 −0.001).
Restricted to what a live system can compute — label-free aggregates over
strictly earlier rows of the entity — it adds nothing over the provider's
own `C*` / `D*` columns, which already summarise the card's past. Kaggle's
gains came from aggregating over the whole dataset (future rows and, through
propagation, the label) on a test set sharing those entities.

**Evidence.** `tests/test_features.py` proves the features depend on no
later row; `docs/feature_sets.md` shows removing `C`/`D`/`M`/`V` costs 0.19
— that is where the history lives.

### "How much of the score is yours?"

The provider's engineered families are worth 0.19 PR-AUC; my frequency
tables and composite keys add 0.02–0.03 wherever they are added; the
largest provider block (`V`, 339 columns) is optional once entity frequency
is modelled — a 100-input model scores the same (`docs/feature_sets.md`,
E020). I would rather say this plainly than claim the model.

### "Depth 12 and 1,600 trees — isn't that overfitting? Your train PR-AUC is 0.99."

The one-axis sweeps (E021) show validation never turning over within the
tested ranges; the deliberately over-capacity model (train 1.000) beats the
anchor on validation, and every regulariser hurt. The gap is mostly
*account memorisation* — labels propagate within accounts in the training
window — which inflates train PR-AUC without harming the ranking of new
rows. The gap is a symptom to watch, not a metric to minimise.

**Strongest objection.** "But it transferred worse on test." It did (+0.017
validation → +0.004 test, operating points slightly worse). I shipped it
because the rule says validation decides and test reports; the right fix
is the validation horizon, not a test-driven pick — `docs/xgboost_progression.md`.

### "Why sigmoid calibration and not isotonic?"

Both fit on 190k out-of-fold training-window scores, never on validation
(ADR 0007). Isotonic was marginally better calibrated but its steps tie
scores and cost 0.011 PR-AUC; sigmoid is strictly monotone, so the ranking
is untouched. Brier 0.0202 → 0.0185, ECE 0.0167 → 0.0037 on validation.

### "Why 0.08? That declines 7 % of transactions."

Because the assumed costs say so: a missed fraud costs the amount + 15, a
false decline 0.10 × amount + 2 (ADR 0006). The amount-weighted cost curve
minimises at 0.08–0.095 with a train-only cross-check at 0.08; the curve is
flat from 0.065 to 0.135, and the sensitivity table moves it between 0.055
and 0.165 under ±50 % cost changes. The *direction* (far below 0.5) is
robust; the value is a business input, stated as such.

**Strongest objection.** "Decline-only is the wrong policy." Agreed:
`docs/review_policy.md` adds a review action (block ≥ 0.42 at the 80 %
precision bar, review the next scores down to the analyst budget) and cuts
validation cost 30 % against decline-only. On test the block precision
drifts 0.80 → 0.71 and a fixed review threshold overshoots its budget, so
reviewing should be rank-based.

### "What fools the model?"

Two things (`docs/error_analysis.md`). Confident false positives are the
fraud archetype — new card, product `C`, no billing address, self-addressed
e-mail, mobile, credit — performed by real customers, row-by-row
indistinguishable: review them, don't block. Confident false negatives are
fraud that looks like everyone else: propagated labels (not recoverable) and
established cards used for the mainstream product (recoverable only via
per-entity deviation on that slice — the targeted experiment not yet run).

### "How do you know production gives the same probability as training?"

One pickled object does feature construction, preprocessing, model and
calibration; the API loads it and, at startup, re-scores 50 frozen rows
through the production input path and refuses to start on any difference or
on an artifact whose sha does not match the golden (`fraud.serve.parity`).
Tests assert training path == artifact == API == golden; a tampered golden
is refused in the container. Model outputs are only reproducible for the
same artifact bytes — the fixture golden pins the preprocessed matrix
instead, because XGBoost on a 260-row fixture builds different trees on
Linux than on macOS (found by CI).

### "What would you do next, and what would you not do?"

Next, in order: a drift-aware validation horizon (ADR before any further
selection); per-product thresholds; per-entity deviation features evaluated
on the `W`/established-card slice; the 100-input no-`V` model if serving
cost matters. Not: ensembles, stacking, pseudo-labels, or another tuning
search — none has a written reason yet (project rule G10).
