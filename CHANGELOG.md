# Changelog

All notable changes to the data pipeline and analysis code are documented here.

## [1.1.0] — Final manuscript version

### Fixed
- **`inflation_rate` construction**: the predictor was previously computed by applying `pct_change()` to the World Bank CPI inflation series, which is already expressed as an annual percentage rate. This double-differencing produced an implausible feature (range approximately ±325, SD≈118). Corrected to use the CPI inflation series directly (range approximately −17 to +557, SD≈9.8, consistent with published inflation statistics). See `src/data/preprocess.py::engineer_features` and the corresponding regression test in `tests/test_data_integrity.py::test_inflation_rate_is_plausible`.
- **Horizon-sensitivity winsorisation window**: an earlier version of the horizon sensitivity analysis (`src/models/horizon_sensitivity.py`) used a winsorisation reference distribution that was constructed independently from the primary 5-year pipeline, producing a 5-year temporal R² inconsistent with the headline model's result. Both analyses now share a single, unified panel-construction function (`src/data/preprocess.py::build_panel`), parametrised only by forecast horizon.
- **LORO bootstrap confidence intervals**: an earlier implementation computed bootstrap CIs for leave-one-region-out validation using GroupKFold out-of-sample predictions restricted to each region's rows, rather than the predictions of the actual region-specific LORO model (trained on all other regions, tested on the held-out region). This caused some reported point estimates to fall outside their own confidence intervals. Fixed in `src/models/validate.py::loro_validation`, which now resamples each LORO model's own (actual, predicted) pairs. See `tests/test_statistical_methods.py::test_bootstrap_ci_preserves_pairing`.
- **Temporal-holdout bootstrap CI pairing**: bootstrap resampling previously resampled `y_true` and `y_pred` arrays independently rather than as matched pairs, which destroys the correlation structure required for valid R² inference and can produce confidence intervals that do not contain the reported point estimate. Fixed in `src/models/validate.py::bootstrap_r2_ci`.

### Changed
- Country fixed-effects (FE) baselines are now evaluated under both GroupKFold and temporal-holdout validation (previously only GroupKFold was reported, based on an incorrect assumption that FE models could not be evaluated under a temporal split).
- Linear regression baseline replaced with Ridge regression (α=1) throughout, to avoid numerical instability from near-collinear standardised features in small cross-validation folds.

### Added
- `tests/test_data_integrity.py` and `tests/test_statistical_methods.py`: regression tests encoding the corrections above, so that any future modification to the pipeline cannot silently reintroduce these defects.
- Feature ablation analysis (`src/models/explain.py::feature_ablation`) with per-feature VIF cross-reference.
- SHAP multi-seed stability check (`src/models/explain.py::shap_multiseed_stability`), computed on independent random subsamples of the full dataset rather than a single fixed subsample.
- Structural break test (`src/models/validate.py::structural_break_test`) for the pre-/post-2015 (COVID-19 / Paris Agreement) period.

## [1.0.0] — Initial pipeline

- Initial data download, feature engineering, model training, and explainability pipeline.
