"""
train.py
========
Trains and evaluates all six ML models plus the baseline hierarchy under
GroupKFold(5) and temporal-holdout validation. Performs Optuna hyperparameter
tuning for XGBoost.

Usage:
    python train.py --data data/cleaned/ml_co2forecast.csv --out-dir results
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
N_TRIALS = 50  # Optuna trials; reduce for a faster (less optimal) run
RANDOM_STATE = 42


def load_data(path: str):
    df = pd.read_csv(path)
    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df["country_code"].values
    years = df["year"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    tr_mask = years <= TEMPORAL_CUTOFF_YEAR
    te_mask = years > TEMPORAL_CUTOFF_YEAR

    return {
        "df": df,
        "X": X_sc,
        "y": y,
        "groups": groups,
        "years": years,
        "scaler": scaler,
        "X_tr": X_sc[tr_mask],
        "y_tr": y[tr_mask],
        "X_te": X_sc[te_mask],
        "y_te": y[te_mask],
    }


def tune_xgboost(X, y, groups, n_trials: int = N_TRIALS) -> dict:
    """Optuna TPE search over XGBoost hyperparameters, optimised for mean
    GroupKFold(5) R²."""
    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("ss", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("cbt", 0.6, 1.0),
            reg_alpha=trial.suggest_float("ra", 1e-8, 10, log=True),
            reg_lambda=trial.suggest_float("rl", 1e-8, 10, log=True),
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        scores = []
        for tr_idx, te_idx in gkf.split(X, y, groups):
            model = XGBRegressor(**params)
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[te_idx])
            scores.append(r2_score(y[te_idx], preds))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials)

    best = dict(study.best_params)
    best["learning_rate"] = best.pop("lr")
    best["subsample"] = best.pop("ss")
    best["colsample_bytree"] = best.pop("cbt")
    best["reg_alpha"] = best.pop("ra")
    best["reg_lambda"] = best.pop("rl")
    best["random_state"] = RANDOM_STATE
    best["verbosity"] = 0
    return best


def get_models(xgb_params: dict):
    return [
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("Random Forest", RandomForestRegressor(n_estimators=300, max_depth=8, n_jobs=-1, random_state=RANDOM_STATE)),
        ("XGBoost", XGBRegressor(**xgb_params)),
        ("LightGBM", LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)),
        ("CatBoost", CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, random_seed=RANDOM_STATE, verbose=0)),
    ]


def evaluate_groupkfold(models, X, y, groups):
    gkf = GroupKFold(n_splits=5)
    rows = []
    for name, model in models:
        r2s, rmses, maes = [], [], []
        for tr_idx, te_idx in gkf.split(X, y, groups):
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[te_idx])
            r2s.append(r2_score(y[te_idx], preds))
            rmses.append(np.sqrt(mean_squared_error(y[te_idx], preds)))
            maes.append(mean_absolute_error(y[te_idx], preds))
        rows.append(
            {
                "Model": name,
                "R2": round(float(np.mean(r2s)), 4),
                "R2_sd": round(float(np.std(r2s)), 4),
                "RMSE": round(float(np.mean(rmses)), 3),
                "MAE": round(float(np.mean(maes)), 3),
            }
        )
    return pd.DataFrame(rows)


def evaluate_temporal(models, X_tr, y_tr, X_te, y_te):
    rows = []
    for name, model in models:
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        rows.append(
            {
                "Model": name,
                "R2_temporal": round(r2_score(y_te, preds), 4),
                "RMSE_temporal": round(np.sqrt(mean_squared_error(y_te, preds)), 3),
                "MAE_temporal": round(mean_absolute_error(y_te, preds), 3),
            }
        )
    return pd.DataFrame(rows)


def evaluate_baselines(X, y, groups, X_tr, y_tr, X_te, y_te, df):
    """Naive, country-FE-only, and country-FE-with-features baselines."""
    gkf = GroupKFold(n_splits=5)
    country_dummies = pd.get_dummies(df["country_code"], drop_first=True).values.astype(float)
    X_fe = np.hstack([X, country_dummies])

    # Naive: predict 0% change
    naive_gkf = [r2_score(y[te], np.zeros(len(te))) for _, te in gkf.split(X, y, groups)]
    naive_temporal = r2_score(y_te, np.zeros(len(y_te)))

    # Country FE only
    fe_only_gkf = []
    for tr, te in gkf.split(X, y, groups):
        m = Ridge(alpha=1.0)
        m.fit(country_dummies[tr], y[tr])
        fe_only_gkf.append(r2_score(y[te], m.predict(country_dummies[te])))

    years = df["year"].values
    tr_mask = years <= TEMPORAL_CUTOFF_YEAR
    te_mask = years > TEMPORAL_CUTOFF_YEAR
    fe_only_temp = Ridge(alpha=1.0)
    fe_only_temp.fit(country_dummies[tr_mask], y[tr_mask])
    fe_only_temporal = r2_score(y[te_mask], fe_only_temp.predict(country_dummies[te_mask]))

    # Country FE + features
    fe_feat_gkf = []
    for tr, te in gkf.split(X, y, groups):
        m = Ridge(alpha=1.0)
        m.fit(X_fe[tr], y[tr])
        fe_feat_gkf.append(r2_score(y[te], m.predict(X_fe[te])))

    fe_feat_temp = Ridge(alpha=1.0)
    fe_feat_temp.fit(X_fe[tr_mask], y[tr_mask])
    fe_feat_temporal = r2_score(y[te_mask], fe_feat_temp.predict(X_fe[te_mask]))

    return pd.DataFrame(
        [
            {"Model": "Naive (predict 0% change)", "GKF_R2": round(np.mean(naive_gkf), 4), "Temporal_R2": round(naive_temporal, 4)},
            {"Model": "Country FE only", "GKF_R2": round(np.mean(fe_only_gkf), 4), "Temporal_R2": round(fe_only_temporal, 4)},
            {"Model": "Country FE + features", "GKF_R2": round(np.mean(fe_feat_gkf), 4), "Temporal_R2": round(fe_feat_temporal, 4)},
        ]
    )


def main(data_path: str, out_dir: str, n_trials: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path("data/processed")
    model_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_data(data_path)

    print("Tuning XGBoost (Optuna TPE)...")
    xgb_params = tune_xgboost(bundle["X"], bundle["y"], bundle["groups"], n_trials=n_trials)
    with open(model_dir / "xgb_params.json", "w") as f:
        json.dump(xgb_params, f, indent=2)
    print(f"Best params: {xgb_params}")

    models = get_models(xgb_params)

    print("\nGroupKFold(5) evaluation...")
    gkf_results = evaluate_groupkfold(models, bundle["X"], bundle["y"], bundle["groups"])
    gkf_results.to_csv(out / "model_comparison_groupkfold.csv", index=False)
    print(gkf_results.to_string(index=False))

    print("\nTemporal holdout evaluation...")
    temp_results = evaluate_temporal(models, bundle["X_tr"], bundle["y_tr"], bundle["X_te"], bundle["y_te"])
    temp_results.to_csv(out / "model_comparison_temporal.csv", index=False)
    print(temp_results.to_string(index=False))

    print("\nBaseline hierarchy evaluation...")
    baselines = evaluate_baselines(
        bundle["X"], bundle["y"], bundle["groups"],
        bundle["X_tr"], bundle["y_tr"], bundle["X_te"], bundle["y_te"],
        bundle["df"],
    )
    baselines.to_csv(out / "baseline_comparison.csv", index=False)
    print(baselines.to_string(index=False))

    # Persist the final XGBoost model (trained on all data) for downstream
    # SHAP/permutation/PDP analysis
    xgb_final = XGBRegressor(**xgb_params)
    xgb_final.fit(bundle["X"], bundle["y"])
    joblib.dump(xgb_final, model_dir / "xgb_final.pkl")
    joblib.dump(bundle["scaler"], model_dir / "scaler.pkl")
    np.save(model_dir / "X_scaled.npy", bundle["X"])
    np.save(model_dir / "y.npy", bundle["y"])
    np.save(model_dir / "groups.npy", bundle["groups"])

    print(f"\nAll results saved to {out}/ and model artifacts to {model_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cleaned/ml_co2forecast.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    main(args.data, args.out_dir, args.n_trials)
