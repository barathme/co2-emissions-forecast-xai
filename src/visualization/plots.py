"""
plots.py
========
Generates all manuscript figures from the saved results/ CSVs and model
artifacts. Each function corresponds to one manuscript figure and writes
a 300 DPI PNG to the figures/ directory.

Usage:
    python plots.py --results-dir results --out-dir figures
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = ["#2C3E50", "#C0392B", "#D68910", "#148F77", "#7D3C98", "#1A5276", "#B03A2E"]
FEATURES = [
    "gdp_growth_calc",
    "log_gdp_pc",
    "internet_penetration",
    "inflation_rate",
    "hdi",
    "pop_growth",
    "relative_co2_intensity",
]
FEATURE_LABELS = [
    "GDP Growth",
    "Log GDP/Cap",
    "Internet\nPenetration",
    "Inflation",
    "HDI",
    "Pop Growth",
    "Rel CO\u2082\nIntensity",
]


def fig_model_comparison(results_dir: Path, out_dir: Path):
    gkf = pd.read_csv(results_dir / "model_comparison_groupkfold.csv")
    temp = pd.read_csv(results_dir / "model_comparison_temporal.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(gkf["Model"], gkf["R2"], yerr=gkf["R2_sd"], color=COLORS[: len(gkf)], alpha=0.85, capsize=4)
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("GroupKFold(5) R\u00b2")
    axes[0].set_title("GroupKFold(5) Model Comparison", fontweight="bold")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right", fontsize=8)

    axes[1].bar(temp["Model"], temp["R2_temporal"], color=COLORS[: len(temp)], alpha=0.85)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel("Temporal Holdout R\u00b2")
    axes[1].set_title("Temporal Holdout (Train\u21922013, Test 2014\u20132018)", fontweight="bold")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_model_comparison.png")
    plt.close()


def fig_shap_analysis(results_dir: Path, out_dir: Path):
    shap_values = np.load("data/processed/shap_values.npy")
    X = np.load("data/processed/X_scaled.npy")
    shap_df = pd.read_csv(results_dir / "shap_importance.csv")
    perm_df = pd.read_csv(results_dir / "permutation_importance.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    order = np.argsort(np.abs(shap_values).mean(0))[::-1]
    labels_sorted = [FEATURE_LABELS[i].replace("\n", " ") for i in order]
    shap_sorted = shap_values[:, order]
    X_sorted = X[:, order]
    for i, col in enumerate(shap_sorted.T):
        norm_val = (X_sorted[:, i] - X_sorted[:, i].min()) / (
            X_sorted[:, i].max() - X_sorted[:, i].min() + 1e-10
        )
        axes[0].scatter(
            col,
            [i] * len(col) + np.random.normal(0, 0.04, len(col)),
            c=norm_val,
            cmap="coolwarm",
            alpha=0.2,
            s=5,
        )
    axes[0].set_yticks(range(len(labels_sorted)))
    axes[0].set_yticklabels(labels_sorted, fontsize=10)
    axes[0].axvline(0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("SHAP Value")
    axes[0].set_title("SHAP Beeswarm (full dataset)", fontweight="bold")

    shap_asc = shap_df.sort_values("mean_abs_shap", ascending=True)
    axes[1].barh(
        range(len(shap_asc)),
        shap_asc["mean_abs_shap"],
        color=[COLORS[i % len(COLORS)] for i in range(len(shap_asc))],
        alpha=0.85,
    )
    axes[1].set_yticks(range(len(shap_asc)))
    axes[1].set_yticklabels(
        [FEATURE_LABELS[FEATURES.index(f)].replace("\n", " ") for f in shap_asc["feature"]], fontsize=10
    )
    axes[1].set_xlabel("Mean |SHAP Value|")
    axes[1].set_title("SHAP Global Importance", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_shap_analysis.png")
    plt.close()


def fig_horizon_sensitivity(results_dir: Path, out_dir: Path):
    hz = pd.read_csv(results_dir / "horizon_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(hz["horizon_years"], hz["gkf_r2"], "o-", color=COLORS[0], lw=2.5, ms=9, label="GroupKFold(5) R\u00b2")
    axes[0].fill_between(
        hz["horizon_years"],
        hz["gkf_r2"] - hz["gkf_r2_sd"],
        hz["gkf_r2"] + hz["gkf_r2_sd"],
        alpha=0.15,
        color=COLORS[0],
    )
    valid = hz.dropna(subset=["temporal_r2"])
    axes[0].plot(valid["horizon_years"], valid["temporal_r2"], "s--", color=COLORS[1], lw=2.5, ms=9, label="Temporal R\u00b2")
    axes[0].set_xlabel("Forecast Horizon (years)")
    axes[0].set_ylabel("XGBoost R\u00b2")
    axes[0].set_title("Forecast Horizon Sensitivity", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].bar(hz["horizon_years"], hz["n_train"], color=COLORS[0], alpha=0.75, label="Train", width=1.4)
    axes[1].bar(hz["horizon_years"], hz["n_test"], bottom=hz["n_train"], color=COLORS[1], alpha=0.75, label="Test", width=1.4)
    axes[1].set_xlabel("Forecast Horizon (years)")
    axes[1].set_ylabel("Observations")
    axes[1].set_title("Sample Split by Horizon", fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_horizon_sensitivity.png")
    plt.close()


def fig_loro(results_dir: Path, out_dir: Path):
    loro = pd.read_csv(results_dir / "loro_validation.csv")
    fig, ax = plt.subplots(figsize=(8, 5.5))

    labels = [r.replace("_", " ")[:22] for r in loro["region"]]
    r2v = loro["r2"].values
    cilo = loro["ci_lower"].values
    cihi = loro["ci_upper"].values
    bar_colors = ["#C0392B" if v < 0 else "#148F77" if v > 0.1 else "#D68910" for v in r2v]
    ax.barh(labels, r2v, color=bar_colors, alpha=0.85)
    for i, (lo, hi) in enumerate(zip(cilo, cihi)):
        ax.plot([lo, hi], [i, i], "-", color="#333", lw=2)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("R\u00b2 on held-out region (+ 95% bootstrap CI)")
    ax.set_title("Leave-One-Region-Out Validation", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_loro.png")
    plt.close()


def fig_pdp(results_dir: Path, out_dir: Path):
    pdp_df = pd.read_csv(results_dir / "pdp.csv")
    ice_df = pd.read_csv(results_dir / "ice.csv")

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    for i, (feat, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
        ax = axes[i]
        sub = pdp_df[pdp_df["feature"] == feat]
        if feat in ice_df["feature"].values:
            ice_sub = ice_df[ice_df["feature"] == feat]
            for obs in list(ice_sub["obs"].unique())[:25]:
                o = ice_sub[ice_sub["obs"] == obs]
                ax.plot(o["feature_value_scaled"], o["predicted"], color="#CCCCCC", lw=0.5, alpha=0.5)
        ax.plot(sub["feature_value_scaled"], sub["predicted"], color=COLORS[i % len(COLORS)], lw=2.5)
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_title(label.replace("\n", " "), fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Standardised value", fontsize=8)
        ax.set_ylabel("Predicted CO\u2082\u03945yr (%)", fontsize=8)
    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_pdp_ice.png")
    plt.close()


def main(results_dir: str, out_dir: str):
    results = Path(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    fig_model_comparison(results, out)
    print("  fig_model_comparison.png")
    fig_shap_analysis(results, out)
    print("  fig_shap_analysis.png")
    if (results / "horizon_sensitivity.csv").exists():
        fig_horizon_sensitivity(results, out)
        print("  fig_horizon_sensitivity.png")
    if (results / "loro_validation.csv").exists():
        fig_loro(results, out)
        print("  fig_loro.png")
    if (results / "pdp.csv").exists():
        fig_pdp(results, out)
        print("  fig_pdp_ice.png")

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()
    main(args.results_dir, args.out_dir)
