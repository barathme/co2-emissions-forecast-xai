"""
validate.py
===========
Statistical validation suite: bootstrap confidence intervals, Diebold-Mariano
predictive accuracy tests, leave-one-region-out (LORO) validation with
bootstrap CIs, and structural break (COVID-19 / Paris Agreement) tests.

Usage:
    python validate.py --data data/cleaned/ml_co2forecast.csv --out-dir results
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
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


def bootstrap_r2_ci(y_true, y_pred, n_iter: int = 2000, seed: int = RANDOM_STATE):
    """Paired observation-level bootstrap with replacement, preserving the
    (actual, predicted) correlation structure required for valid R²
    inference."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        boots.append(r2_score(y_true[idx], y_pred[idx]))
    boots = np.array(boots)
    return {
        "mean": float(boots.mean()),
        "ci_lower": float(np.percentile(boots, 2.5)),
        "ci_upper": float(np.percentile(boots, 97.5)),
    }


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, hac_lag: int = 1) -> dict:
    """
    Diebold-Mariano test comparing squared-error loss of two forecasters.
    Positive statistic indicates model 1 (e1) has lower loss than model 2 (e2).
    """
    d = e2**2 - e1**2
    n = len(d)
    d_bar = d.mean()
    gamma0 = np.mean((d - d_bar) ** 2)
    gammas = sum(
        np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)) for lag in range(1, hac_lag + 1)
    ) if hac_lag > 0 else 0.0
    var_d = (gamma0 + 2 * gammas) / n
    dm_stat = d_bar / np.sqrt(max(var_d, 1e-12))
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return {"dm_statistic": float(dm_stat), "p_value": float(p_value)}


def loro_validation(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, xgb_params: dict, n_boot: int = 1000):
    """Leave-one-region-out validation. Bootstrap CIs are computed from each
    region's OWN held-out-model predictions (not a different validation
    strategy's predictions) -- this distinction matters for correctness."""
    rows = []
    rng = np.random.RandomState(RANDOM_STATE)
    for region in df["world_6region"].dropna().unique():
        train_mask = (df["world_6region"] != region).values
        test_mask = (df["world_6region"] == region).values
        if test_mask.sum() < 20:
            continue

        model = XGBRegressor(**xgb_params)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])

        r2 = r2_score(y[test_mask], preds)
        rmse = np.sqrt(mean_squared_error(y[test_mask], preds))

        y_region = y[test_mask]
        boots = []
        n_region = test_mask.sum()
        for _ in range(n_boot):
            idx = rng.choice(n_region, n_region, replace=True)
            boots.append(r2_score(y_region[idx], preds[idx]))

        rows.append(
            {
                "region": region,
                "n_test": int(n_region),
                "r2": round(r2, 4),
                "ci_lower": round(np.percentile(boots, 2.5), 4),
                "ci_upper": round(np.percentile(boots, 97.5), 4),
                "rmse": round(rmse, 3),
            }
        )
    return pd.DataFrame(rows)


def structural_break_test(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, groups: np.ndarray, xgb_params: dict):
    """Welch t-test for pre-/post-2015 target mean shift, plus GroupKFold R²
    by origin-year cohort (to check for systematic prediction bias around
    the COVID-19 / Paris Agreement structural break)."""
    pre_mask = df["year"].values < 2015
    post_mask = df["year"].values >= 2015

    t_stat, p_val = ttest_ind(y[pre_mask], y[post_mask])

    gkf = GroupKFold(n_splits=5)
    preds = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        model = XGBRegressor(**xgb_params)
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])

    r2_pre = r2_score(y[pre_mask], preds[pre_mask])
    r2_post = r2_score(y[post_mask], preds[post_mask])

    return {
        "pre_2015_mean_target": float(y[pre_mask].mean()),
        "post_2015_mean_target": float(y[post_mask].mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "gkf_r2_pre_2015": round(r2_pre, 4),
        "gkf_r2_post_2015": round(r2_post, 4),
    }


def main(data_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    model_dir = Path("data/processed")
    xgb = joblib.load(model_dir / "xgb_final.pkl")
    X = np.load(model_dir / "X_scaled.npy")
    y = np.load(model_dir / "y.npy")
    groups = np.load(model_dir / "groups.npy", allow_pickle=True)
    with open(model_dir / "xgb_params.json") as f:
        xgb_params = json.load(f)

    years = df["year"].values
    tr_mask = years <= TEMPORAL_CUTOFF_YEAR
    te_mask = years > TEMPORAL_CUTOFF_YEAR

    print("Bootstrap CI on temporal holdout R² (XGBoost)...")
    model_temp = XGBRegressor(**xgb_params)
    model_temp.fit(X[tr_mask], y[tr_mask])
    preds_temp = model_temp.predict(X[te_mask])
    ci = bootstrap_r2_ci(y[te_mask], preds_temp)
    print(f"  R²={r2_score(y[te_mask], preds_temp):.4f}, 95% CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    with open(out / "bootstrap_ci_temporal.json", "w") as f:
        json.dump(ci, f, indent=2)

    print("\nLORO validation...")
    loro = loro_validation(df, X, y, xgb_params)
    loro.to_csv(out / "loro_validation.csv", index=False)
    print(loro.to_string(index=False))

    print("\nStructural break test...")
    break_test = structural_break_test(df, X, y, groups, xgb_params)
    with open(out / "structural_break_test.json", "w") as f:
        json.dump(break_test, f, indent=2)
    print(json.dumps(break_test, indent=2))

    print(f"\nAll validation results saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cleaned/ml_co2forecast.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    main(args.data, args.out_dir)
