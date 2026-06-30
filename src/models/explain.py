"""
explain.py
==========
Consistent feature attribution: SHAP, permutation importance, partial
dependence (PDP), individual conditional expectation (ICE), and feature
ablation -- all computed from the single trained XGBoost model to avoid
the attribution-inconsistency pitfall common in prior literature (i.e.
explaining one model's predictions using a different model's SHAP values).

Usage:
    python explain.py --data data/cleaned/ml_co2forecast.csv --out-dir results
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

FEATURES = [
    "gdp_growth_calc",
    "log_gdp_pc",
    "internet_penetration",
    "inflation_rate",
    "hdi",
    "pop_growth",
    "relative_co2_intensity",
]
TARGET = "co2_delta_5yr_pct"
TEMPORAL_CUTOFF_YEAR = 2013
RANDOM_STATE = 42


def compute_shap(model, X: np.ndarray) -> np.ndarray:
    """SHAP values on the FULL dataset (not a subsample) using TreeExplainer,
    which is exact (not approximate) for tree-based models."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def shap_multiseed_stability(model, X: np.ndarray, seeds=(42, 123, 456, 789, 1001), n_sample: int = 1000):
    """Cross-subsample SHAP stability check: compute mean |SHAP| on several
    independent random subsamples and report the coefficient of variation
    across seeds. This is descriptive (per Strumbelj & Kononenko, 2014);
    no universal CV threshold is asserted as a formal stability criterion."""
    explainer = shap.TreeExplainer(model)
    results = []
    for seed in seeds:
        idx = np.random.RandomState(seed).choice(len(X), n_sample, replace=False)
        sv = explainer.shap_values(X[idx])
        results.append(np.abs(sv).mean(0))
    arr = np.array(results)
    return arr.mean(0), arr.std(0)


def compute_permutation_importance(model, X: np.ndarray, y: np.ndarray, n_repeats: int = 20):
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE, scoring="r2"
    )
    return result.importances_mean, result.importances_std


def compute_pdp_ice(model, X: np.ndarray, feature_idx: int, n_grid: int = 60, n_ice: int = 25):
    """Partial dependence (mean marginal effect) and individual conditional
    expectation (ICE) curves for a single feature."""
    feature_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_grid)
    median_row = np.median(X, axis=0)

    pdp = []
    grid = np.tile(median_row, (n_grid, 1))
    grid[:, feature_idx] = feature_values
    pdp = model.predict(grid)

    rng = np.random.RandomState(RANDOM_STATE)
    ice_idx = rng.choice(len(X), n_ice, replace=False)
    ice_curves = {}
    for obs_i in ice_idx:
        row = X[obs_i].copy()
        ice_grid = np.tile(row, (n_grid, 1))
        ice_grid[:, feature_idx] = feature_values
        ice_curves[int(obs_i)] = model.predict(ice_grid).tolist()

    return feature_values, pdp, ice_curves


def feature_ablation(X_tr, y_tr, X_te, y_te, xgb_params: dict):
    """Drop each feature in turn, retrain, and measure the temporal-holdout
    R² delta relative to the full-feature baseline."""
    baseline_model = XGBRegressor(**xgb_params)
    baseline_model.fit(X_tr, y_tr)
    baseline_r2 = r2_score(y_te, baseline_model.predict(X_te))

    rows = []
    for i, feat in enumerate(FEATURES):
        X_tr_ab = np.delete(X_tr, i, axis=1)
        X_te_ab = np.delete(X_te, i, axis=1)
        model = XGBRegressor(**xgb_params)
        model.fit(X_tr_ab, y_tr)
        r2 = r2_score(y_te, model.predict(X_te_ab))
        rows.append(
            {
                "feature_dropped": feat,
                "temporal_r2": round(r2, 4),
                "temporal_delta": round(r2 - baseline_r2, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("temporal_delta"), baseline_r2


def main(data_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path("data/processed")

    df = pd.read_csv(data_path)
    xgb = joblib.load(model_dir / "xgb_final.pkl")
    X = np.load(model_dir / "X_scaled.npy")
    y = np.load(model_dir / "y.npy")
    with open(model_dir / "xgb_params.json") as f:
        xgb_params = json.load(f)

    print("Computing SHAP values (full dataset)...")
    shap_values = compute_shap(xgb, X)
    mean_abs_shap = np.abs(shap_values).mean(0)
    shap_df = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False
    )
    shap_df.to_csv(out / "shap_importance.csv", index=False)
    np.save(model_dir / "shap_values.npy", shap_values)
    print(shap_df.to_string(index=False))

    print("\nSHAP multi-seed stability check...")
    mean_ms, std_ms = shap_multiseed_stability(xgb, X)
    cv_pct = std_ms / (mean_ms + 1e-9) * 100
    stability_df = pd.DataFrame(
        {"feature": FEATURES, "mean_abs_shap": mean_ms, "sd_across_seeds": std_ms, "cv_pct": cv_pct}
    ).sort_values("mean_abs_shap", ascending=False)
    stability_df.to_csv(out / "shap_stability.csv", index=False)
    print(stability_df.to_string(index=False))

    print("\nPermutation importance...")
    perm_mean, perm_std = compute_permutation_importance(xgb, X, y)
    perm_df = pd.DataFrame(
        {"feature": FEATURES, "perm_importance": perm_mean, "perm_importance_sd": perm_std}
    ).sort_values("perm_importance", ascending=False)
    perm_df.to_csv(out / "permutation_importance.csv", index=False)
    print(perm_df.to_string(index=False))

    print("\nPDP + ICE for all features...")
    pdp_rows, ice_rows = [], []
    for i, feat in enumerate(FEATURES):
        fv, pdp, ice = compute_pdp_ice(xgb, X, i)
        for f, p in zip(fv, pdp):
            pdp_rows.append({"feature": feat, "feature_value_scaled": f, "predicted": p})
        for obs_id, curve in ice.items():
            for f, p in zip(fv, curve):
                ice_rows.append({"feature": feat, "obs": obs_id, "feature_value_scaled": f, "predicted": p})
    pd.DataFrame(pdp_rows).to_csv(out / "pdp.csv", index=False)
    pd.DataFrame(ice_rows).to_csv(out / "ice.csv", index=False)
    print("PDP/ICE saved.")

    print("\nFeature ablation (temporal holdout)...")
    years = df["year"].values
    tr_mask = years <= TEMPORAL_CUTOFF_YEAR
    te_mask = years > TEMPORAL_CUTOFF_YEAR
    ablation_df, baseline_r2 = feature_ablation(X[tr_mask], y[tr_mask], X[te_mask], y[te_mask], xgb_params)
    ablation_df.to_csv(out / "feature_ablation.csv", index=False)
    print(f"Baseline temporal R²={baseline_r2:.4f}")
    print(ablation_df.to_string(index=False))

    print(f"\nAll explainability outputs saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cleaned/ml_co2forecast.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.data, args.out_dir)
