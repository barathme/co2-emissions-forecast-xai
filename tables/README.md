# Tables

Final, validated tables matching the published manuscript. These are static snapshots; running the full pipeline (`scripts/run_full_pipeline.py`) regenerates equivalent files in `results/`.

| File | Manuscript Table | Description |
|---|---|---|
| `table02_vif.csv` | Table 2 | Variance inflation factors for all seven predictors |
| `table03_model_comparison_groupkfold.csv` | Table 3 | GroupKFold(5) R², RMSE, MAE for all models |
| `table03b_baseline_hierarchy.csv` | Table 3 (baselines) | Naive, country FE-only, country FE+features baselines |
| `table04_temporal_holdout.csv` | Table 4 | Temporal holdout R², RMSE, MAE for all models |
| `table06_feature_ablation.csv` | Table 6 | Feature ablation: temporal R² impact of dropping each predictor |
| `table07_shap_importance.csv` | Table 7 | SHAP global feature importance |
| `table07b_permutation_importance.csv` | Table 7 | Permutation importance (cross-checked against SHAP) |
| `table08_horizon_sensitivity.csv` | Table 8 | Forecast horizon sensitivity (3/5/7/10 years) |
| `table09_loro_validation.csv` | Table 9 | Leave-one-region-out R² and bootstrap 95% CIs |
| `table10_equity_by_income.csv` | Table 10 | Prediction equity (R², RMSE, residual) by World Bank income group |

All numeric values in these tables match the corresponding figures and discussion in the manuscript text.
