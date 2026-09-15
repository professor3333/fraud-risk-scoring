# Testing

`uv run pytest` — 93 tests on a synthetic 400-row fixture (`tests/fixtures/`),
no data download, no network, ~15 s. `uv run pytest -m slow` — 4 tests
against the real files and the production artifact when present. CI runs
the default set plus a smoke training through the CLI, a Docker build and a
container health check (`.github/workflows/ci.yml`).

## ML-specific properties and the test that owns each

| property | test |
|---|---|
| schema (columns, dtypes, non-null set) | `test_data::test_transaction_columns_and_shape`, `test_identity_columns_and_shape`, `test_string_columns_have_string_dtype`, `test_validate_rejects_missing_column` |
| join cardinality (every transaction kept, none duplicated, orphans rejected) | `test_data::test_join_keeps_every_transaction_and_flags_identity`, `test_join_rejects_orphan_identity_rows`, `test_validate_join_catches_duplication_and_wrong_flag` |
| target values ∈ {0, 1}, rate in band | `test_data::test_label_is_binary_and_rare`, `test_validate_rejects_label_outside_binary` |
| temporal split (windows ordered, disjoint, sized from config) | `test_split::test_windows_are_strictly_later_and_disjoint`, `test_assign_window_matches_day_boundaries`, `test_config_windows_are_ordered_and_cover_the_data` |
| train before validation | `test_ml_contracts::test_train_is_strictly_before_validation` |
| validation before test | `test_ml_contracts::test_validation_is_strictly_before_test` |
| tuning folds inside the training window | `test_split::test_tuning_folds_stay_inside_training_window`, `test_fold_frames_are_strictly_ordered` |
| no target / time / id in features | `test_pipeline::test_feature_spec_excludes_forbidden_columns` (spec), `test_ml_contracts::test_fitted_pipeline_never_reads_target_time_or_id` (fitted object) |
| transformers fit only on training rows | `test_model::test_nothing_is_fit_on_validation_or_test` (corrupt every later row → identical imputer, vocabularies, coefficients), `test_features::test_frequency_encoder_learns_from_fit_rows_only` |
| time-aware features use strictly earlier rows | `test_features::test_entity_history_uses_only_strictly_earlier_rows`, `test_entity_history_is_unaffected_by_later_rows`, `test_all_derivers_are_row_local` |
| unknown category (and missing identity, rare levels) | `test_pipeline::test_unseen_categories_and_missing_identity`, `test_rare_levels_and_unseen_levels_share_one_column`, `test_features::test_frequency_encoder_learns_from_fit_rows_only` (unseen → 0) |
| feature count fixed after fit, identical across windows | `test_ml_contracts::test_feature_count_is_fixed_after_fit`, `test_pipeline::test_fit_transform_shapes_are_stable` |
| prediction shape | `test_ml_contracts::test_prediction_shape_and_range` |
| prediction range [0, 1] | `test_ml_contracts::test_prediction_shape_and_range`, `test_model::test_probabilities_in_unit_interval` |
| model beats the no-skill baseline | `test_model::test_logreg_beats_constant_on_validation` |
| reproducibility (same config + seed → same metric) | `test_model::test_run_is_reproducible` |
| model serialization (save → load → identical, twice) | `test_ml_contracts::test_model_serialization_roundtrip_is_exact`, `test_pipeline::test_save_load_identical_predictions` |
| training / inference parity | `test_parity::test_training_path_equals_loaded_artifact_raw_scores`, `test_loaded_artifact_matches_frozen_golden`, `test_api_matches_frozen_golden_and_offline`, `test_service_refuses_to_start_on_parity_mismatch`, `test_fixture_feature_matrix_matches_committed_golden`; slow: `test_production_artifact_parity` |
| API contract (routes, required fields, enums, bounds, status codes) | `test_ml_contracts::test_api_contract_via_openapi`, `test_serving::test_bad_input_is_rejected_with_a_useful_body`, `test_model_info`, `test_batch_rejects_empty_and_bad_rows` |
| API == offline, batch == single, CSV == single | `test_serving::test_predict_matches_offline_pipeline`, `test_batch_is_ranked_and_matches_single_predictions`, `test_csv_upload_scores_ranks_and_summarises` |
| calibration preserves ranking; cost / policy arithmetic | `test_model::test_calibrated_model_keeps_ranking_and_improves_brier`, `test_cost_curve_prefers_catching_expensive_fraud`, `test_policy_bands_and_budget_sizing`, `test_top_k_per_day_reviews_the_highest_scores` |
| experiment record complete (provenance, metrics, artifacts) | `test_model::test_run_logs_provenance_and_artifacts` |

## Fixture

`tests/fixtures/make_fixtures.py` generates 400 synthetic transactions and 100
identity rows with the real schema, spread over all 183 days, with positives
in every split window and a weak planted signal (product `C`, larger
amounts) so "beats the baseline" is testable. `--golden` re-fits the fixture
pipeline from the committed config and stores the **preprocessed feature
matrix** for a frozen sample (`tests/fixtures/frozen_features.json`), which
`test_parity` compares against on every platform. Model *outputs* are not
part of that golden on purpose: XGBoost trees fitted on a 260-row fixture
differ between macOS and Linux runners (found when CI first ran it —
0.192 vs 0.059 for one row), so probabilities are pinned per artifact by
`scripts/freeze_artifact.py` and checked by the same bytes that serve them.
Regenerate the fixture golden only when a change to preprocessing is
intended.

## CI

```
push / PR → ruff check → ruff format --check → mypy → pytest (fixture)
          → smoke training through the CLI on the fixture (train + calibrate stand-in)
          → docker build → run the container, /health must answer with parity_rows
```

The full IEEE model is never trained in CI; every step runs on the fixture
in under two minutes. The Docker build uses a fixture-trained stand-in
artifact (`scripts/make_fixture_artifact.py`) so the Dockerfile and the
startup parity check are exercised without the competition data.
