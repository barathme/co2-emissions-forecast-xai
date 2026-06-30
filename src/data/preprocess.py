"""
preprocess.py
=============
Builds the analysis-ready panel dataset from the raw downloaded files:
merges all six sources on (country_code, year), engineers the seven
predictor variables, constructs the 5-year forward CO2 change target,
and applies winsorisation.

This module is the single source of truth for feature construction.
All downstream analyses (model training, horizon sensitivity, ablation)
call into this module to guarantee a consistent, leakage-free dataset.

Usage:
    python preprocess.py --raw-dir data/raw --out-dir data/cleaned
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def load_and_merge(raw_dir: Path) -> pd.DataFrame:
    """Load all raw sources and merge into a single country-year panel."""
    owid = pd.read_csv(raw_dir / "owid_co2_macro.csv", low_memory=False)
    owid = owid.rename(columns={"iso_code": "country_code"})
    owid = owid[
        [
            "country_code",
            "year",
            "co2_per_capita",
            "co2",
            "primary_energy_consumption",
            "population",
            "gdp",
        ]
    ].rename(columns={"gdp": "gdp_usd"})

    gdp = pd.read_csv(raw_dir / "gdp.csv")
    pop = pd.read_csv(raw_dir / "population.csv")
    cpi = pd.read_csv(raw_dir / "cpi_inflation.csv")
    internet = pd.read_csv(raw_dir / "gapminder_internet.csv")
    hdi = pd.read_csv(raw_dir / "gapminder_hdi.csv")

    # NOTE: exact column names depend on the specific Datahub/Gapminder
    # snapshot in use; downstream code expects: country_code, year,
    # gdp_per_capita, inflation_cpi, internet_penetration, hdi.
    # See data/raw/README.md for the expected schema of each source.

    df = owid.copy()
    df = df.merge(gdp, on=["country_code", "year"], how="left", suffixes=("", "_wb"))
    df = df.merge(cpi, on=["country_code", "year"], how="left")
    df = df.merge(internet, on=["country_code", "year"], how="left")
    df = df.merge(hdi, on=["country_code", "year"], how="left")

    df["gdp_per_capita"] = df["gdp_usd"] / df["population"].replace(0, np.nan)
    df["co2_intensity"] = df["co2"] / df["primary_energy_consumption"].replace(0, np.nan)

    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Construct the seven predictors (measured at time t) and the target
    (5-year forward CO2 per capita % change, measured at t+horizon).

    Critical correctness note: `inflation_rate` is the CPI inflation
    series AS REPORTED (an annual percentage rate). It must NOT be
    passed through pct_change() again -- doing so double-differences
    an already-percentage series and produces an invalid feature. This
    was identified as a preprocessing defect during manuscript
    preparation and is fixed here permanently.
    """
    df = df.copy()

    df["future_co2_pc"] = df.groupby("country_code")["co2_per_capita"].shift(-horizon)
    df[TARGET] = (
        (df["future_co2_pc"] - df["co2_per_capita"])
        / df["co2_per_capita"].replace(0, np.nan)
        * 100
    )

    df["gdp_growth_calc"] = df.groupby("country_code")["gdp_usd"].pct_change() * 100
    df["pop_growth"] = df.groupby("country_code")["population"].pct_change() * 100
    df["log_gdp_pc"] = np.log1p(df["gdp_per_capita"].clip(lower=1))

    # Correct: use the CPI inflation series directly (already a rate).
    df["inflation_rate"] = df["inflation_cpi"]

    global_co2_intensity = df.groupby("year")["co2_intensity"].transform("mean")
    df["relative_co2_intensity"] = df["co2_intensity"] / global_co2_intensity.replace(0, np.nan)

    return df


def build_panel(
    raw_dir: Path,
    horizon: int = 5,
    origin_year_min: int = 2001,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> pd.DataFrame:
    """End-to-end build: merge raw sources, engineer features, filter, winsorise."""
    df = load_and_merge(raw_dir)
    df = engineer_features(df, horizon=horizon)

    max_origin_year = 2023 - horizon
    df = df[(df["year"] >= origin_year_min) & (df["year"] <= max_origin_year)]
    df = df.dropna(subset=[TARGET] + FEATURES).reset_index(drop=True)

    q_low, q_high = df[TARGET].quantile([winsor_lower, winsor_upper])
    df = df[(df[TARGET] >= q_low) & (df[TARGET] <= q_high)].reset_index(drop=True)

    return df


def main(raw_dir: str, out_dir: str, horizon: int):
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel = build_panel(raw, horizon=horizon)
    out_path = out / "ml_co2forecast.csv"
    panel.to_csv(out_path, index=False)

    print(f"Built panel: n={len(panel)}, countries={panel['country_code'].nunique()}")
    print(f"Years: {panel['year'].min()}-{panel['year'].max()}")
    print(f"Target mean={panel[TARGET].mean():.2f}, std={panel[TARGET].std():.2f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/cleaned")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in years")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.horizon)
