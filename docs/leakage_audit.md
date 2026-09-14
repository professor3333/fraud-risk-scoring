# Leakage audit

One row per feature family. A family enters a model only after its row says
**in**. "Known at authorization" means the value exists before the approve /
decline decision on that transaction. "Fit scope" names what, if anything, is
learned from data and on which rows (G2: training window only).

Provider-computed columns (`C*`, `D*`, `M*`, `V*`, `id_*`) are accepted as
point-in-time on the competition host's statement that they describe the
transaction as it was scored; this is an assumption recorded in ADR 0001, not
something we can verify from the anonymised data.

| family | columns | known at authorization? | uses later rows? | fit scope | verdict | argument |
|---|---|---|---|---|---|---|
| `TransactionID` | 1 | yes | no | none | **out** | Row identifier; monotone in time, so it encodes position in the dataset, not fraud. |
| `TransactionDT` (raw) | 1 | yes | no | none | **out** | Seconds since an undisclosed origin — encodes *which part of the data* the row is in (`docs/eda.md` §2). Consumed only by the split module. Derived cyclic features (hour of day) get their own row when proposed. |
| `TransactionAmt` | 1 | yes | no | median (train), scale (train) | **in** | Amount of the transaction being decided. |
| `ProductCD` | 1 | yes | no | one-hot vocabulary (train) | **in** | Product code of the transaction. |
| `card1` – `card6` | 6 | yes | no | `card1/2/3/5`: median + scale (train); `card4/6`: one-hot (train) | **in** (baseline) | Card attributes known at authorization. `card1` is a 13.5k-level identifier used as a raw number here — a weak, arbitrary encoding, kept only because the baseline is meant to be naive. Any smarter encoding (frequency, target) is a Stage 3 ADR because *how* it is fit decides whether it leaks. |
| `addr1`, `addr2` | 2 | yes | no | median + scale (train) | **in** | Billing region codes. Numeric treatment is arbitrary; same caveat as `card1`. |
| `dist1`, `dist2` | 2 | yes | no | median + scale (train) | **in** | Distances between transaction attributes; 60 – 94 % null, indicator kept. |
| `P_emaildomain`, `R_emaildomain` | 2 | yes | no | — | **out** (baseline) | 59 / 60 levels; presence alone is a strong, mostly product-driven signal. Encoding strategy (grouping, frequency, one-hot with min frequency) is a Stage 3 decision; excluded from the baseline so the baseline stays naive. |
| `C1` – `C14` | 14 | assumed | provider-computed | median + scale (train) | **in** | Counts (e.g. addresses linked to the card) as of the transaction, per the host. Never null. If a later experiment shows implausible separation, revisit. |
| `D1` – `D15` | 15 | assumed | provider-computed | median + scale (train) | **in** (raw values only) | Time deltas to earlier events on the card — past information. **Not** combined with `TransactionDT` here. The `(card1, day − D1)` entity reconstruction is powerful because the label propagates within an entity; it is **out** until ADR 0004 argues it in with a strictly-earlier-rows computation and a test on a synthetic frame. |
| `M1` – `M9` | 9 | assumed | provider-computed | one-hot incl. `<missing>` (train) | **in** | Match flags (e.g. name on card vs address). Missingness is informative and kept as a level. |
| `V1` – `V339` | 339 | assumed | provider-computed | median + scale + missing indicator (train) | **in** | Vesta-engineered features with 15 structured null blocks (`docs/eda.md` §4). Accepted as point-in-time on the host's statement. The imputer keeps all-null columns as zeros so serving-time shape is stable. |
| `id_01` – `id_38`, `DeviceType`, `DeviceInfo` | 40 | yes (when the identity record exists) | no | numeric: median + scale (train); low-card strings: one-hot (train) | **in** except `DeviceInfo`, `id_23`, `id_27`, `id_30`, `id_31`, `id_33`, `id_34` | Device / network attributes captured at the transaction. High-cardinality strings (browser, OS, screen, device model) wait for the Stage 3 encoding ADR. |
| `has_identity` | 1 | yes | no | none (derived from the join) | **in** | Whether an identity record accompanies the transaction. Coverage moves with time and product (`docs/eda.md` §3); legitimate because the serving contract makes the identity record part of the request, and the temporal split measures whether the signal transfers. |
| `isFraud` | 1 | — | — | — | **target** | Never an input; `fraud.features.columns.load_feature_spec` raises if a spec lists it. |

## Preprocessing steps

| step | fit on | applied to | verdict |
|---|---|---|---|
| median imputation (`SimpleImputer`) | training window | all windows, serving | in — `tests/test_model.py::test_nothing_is_fit_on_validation_or_test` |
| missing indicators | training window (which columns) | all | in |
| standard scaling | training window | all | in |
| one-hot vocabularies | training window | all; unseen levels → all-zero row | in |
| temporal split | config only | — | in — `tests/test_split.py` |

## Open items (must be decided before the feature exists)

- ADR 0004: entity reconstruction from `D1` and per-entity rolling aggregates.
- Stage 3 encoding ADR: `card1`, `addr1`, email domains, `DeviceInfo`, `id_30/31/33` — frequency vs target vs grouped one-hot, and on which rows the encoding is fit.
- Hour-of-day from `TransactionDT`: legitimate in principle (the clock is known at authorization); needs its own row and an experiment.
