#!/usr/bin/env python3
"""
run_full_pipeline.py
=====================
End-to-end reproduction script: downloads data, builds the analysis panel,
trains all models, runs statistical validation (bootstrap CIs, Diebold-Mariano,
LORO, structural break test), computes explainability outputs (SHAP,
permutation importance, PDP/ICE, ablation), runs forecast horizon
sensitivity, and generates all figures.

Usage:
    python scripts/run_full_pipeline.py [--skip-download] [--n-trials 50]

Run from the repository root. Approximate runtime: 20-30 minutes on a
standard laptop CPU (the Optuna search over 50 trials is the dominant
cost; pass --n-trials 10 for a much faster, less-optimal run).
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list, label: str):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nFAILED: {label}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="Skip raw data download (use existing files in data/raw/)")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna hyperparameter search trials")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_download:
        run(
            [py, "src/data/download.py", "--output-dir", "data/raw"],
            "STEP 1/6: Downloading raw datasets",
        )
    else:
        print("Skipping download (--skip-download); using existing data/raw/ files.")

    run(
        [py, "src/data/preprocess.py", "--raw-dir", "data/raw", "--out-dir", "data/cleaned"],
        "STEP 2/6: Building analysis panel (feature engineering, leakage-free target construction)",
    )

    run(
        [
            py, "src/models/train.py",
            "--data", "data/cleaned/ml_co2forecast.csv",
            "--out-dir", "results",
            "--n-trials", str(args.n_trials),
        ],
        "STEP 3/6: Training models (GroupKFold, temporal holdout, baseline hierarchy, Optuna HPO)",
    )

    run(
        [py, "src/models/validate.py", "--data", "data/cleaned/ml_co2forecast.csv", "--out-dir", "results"],
        "STEP 4/6: Statistical validation (bootstrap CIs, LORO, structural break test)",
    )

    run(
        [py, "src/models/explain.py", "--data", "data/cleaned/ml_co2forecast.csv", "--out-dir", "results"],
        "STEP 5/6: Explainability (SHAP, permutation importance, PDP/ICE, feature ablation)",
    )

    run(
        [
            py, "src/models/horizon_sensitivity.py",
            "--raw-dir", "data/raw",
            "--out-dir", "results",
            "--xgb-params", "data/processed/xgb_params.json",
        ],
        "STEP 6a/6: Forecast horizon sensitivity",
    )

    run(
        [py, "src/visualization/plots.py", "--results-dir", "results", "--out-dir", "figures"],
        "STEP 6b/6: Generating figures",
    )

    print(f"\n{'=' * 70}\nPipeline complete. Results in results/, figures in figures/.\n{'=' * 70}")


if __name__ == "__main__":
    main()
