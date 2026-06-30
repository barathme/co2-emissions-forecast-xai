"""
horizon_sensitivity.py
=======================
Tests forecast horizon sensitivity (3, 5, 7, 10 years) using a FIXED
2013 train/test cutoff applied consistently across all horizons. This
fixed-cutoff design is essential for valid cross-horizon comparison --
using a different cutoff per horizon (e.g. a floating percentile-based
split) produces non-comparable results across rows.

Usage:
    python horizon_sensitivity.py --raw-dir data/raw --out-dir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.preprocess import build_panel, FEATURES, TARGET  # noqa: E402

TEMPORAL_CUTOFF_YEAR = 2013
HORIZONS = [3, 5, 7, 10]


def main(raw_dir: str, out_dir: str, xgb_params_path: str):
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(xgb_params_path) as f:
        xgb_params = json.load(f)

    rows = []
    for h in HORIZONS:
        panel = build_panel(raw, horizon=h)
        X = panel[FEATURES].values
        y = panel[TARGET].values
        groups = panel["country_code"].values
        years = panel["year"].values

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        tr_mask = years <= TEMPORAL_CUTOFF_YEAR
        te_mask = years > TEMPORAL_CUTOFF_YEAR

        gkf = GroupKFold(n_splits=5)
        r2s = [
            r2_score(
                y[te],
                XGBRegressor(**xgb_params).fit(X_sc[tr], y[tr]).predict(X_sc[te]),
            )
            for tr, te in gkf.split(X_sc, y, groups)
        ]

        if te_mask.sum() > 30:
            model = XGBRegressor(**xgb_params)
            model.fit(X_sc[tr_mask], y[tr_mask])
            r2_temporal = r2_score(y[te_mask], model.predict(X_sc[te_mask]))
        else:
            r2_temporal = None

        rows.append(
            {
                "horizon_years": h,
                "n_total": len(panel),
                "n_train": int(tr_mask.sum()),
                "n_test": int(te_mask.sum()),
                "gkf_r2": round(float(np.mean(r2s)), 4),
                "gkf_r2_sd": round(float(np.std(r2s)), 4),
                "temporal_r2": round(r2_temporal, 4) if r2_temporal is not None else None,
            }
        )
        print(
            f"H={h}yr: n={len(panel)} | GKF R²={np.mean(r2s):.4f}±{np.std(r2s):.4f} "
            f"| Temporal R²={r2_temporal if r2_temporal is not None else 'N/A'}"
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out / "horizon_sensitivity.csv", index=False)
    print(f"\nSaved to {out / 'horizon_sensitivity.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--xgb-params", default="data/processed/xgb_params.json")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.xgb_params)
