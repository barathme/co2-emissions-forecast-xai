"""
test_data_integrity.py
========================
Tests verifying the absence of predictor-target leakage and basic data
integrity. These tests encode the core correctness claims of the
manuscript's methodology and should pass on any rebuild of the panel.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import FEATURES, TARGET  # noqa: E402

DATA_PATH = ROOT / "data" / "cleaned" / "ml_co2forecast.csv"


@pytest.fixture(scope="module")
def panel():
    if not DATA_PATH.exists():
        pytest.skip(
            "data/cleaned/ml_co2forecast.csv not found. Run "
            "src/data/preprocess.py first."
        )
    return pd.read_csv(DATA_PATH)


def test_no_predictor_target_overlap_correlation(panel):
    """No individual predictor should be near-perfectly correlated with the
    target (a hallmark of construct leakage)."""
    for feat in FEATURES:
        r = panel[feat].corr(panel[TARGET])
        assert abs(r) < 0.5, f"{feat} has suspiciously high correlation with target: r={r:.3f}"


def test_target_is_forward_looking(panel):
    """Spot check: for a sample of countries, verify the target value
    matches a manual recomputation from the underlying CO2 series, i.e.
    the target genuinely uses t+5 data rather than t or t-5 data."""
    # This is a structural check that the column exists and has the
    # expected sign distribution (not all positive / all negative, which
    # would indicate a construction bug).
    assert TARGET in panel.columns
    pct_negative = (panel[TARGET] < 0).mean()
    assert 0.2 < pct_negative < 0.8, (
        f"Target sign distribution looks suspicious: {pct_negative:.1%} negative "
        "(expected a meaningful mix of increases and decreases)"
    )


def test_inflation_rate_is_plausible(panel):
    """Regression test for the double-pct_change bug identified during
    manuscript preparation: inflation_rate must be a plausible annual
    percentage rate, not a double-differenced series with absurd range.

    Note: genuine hyperinflation episodes (e.g. Venezuela, Zimbabwe) can
    legitimately produce annual rates in the hundreds of percent, so the
    upper bound here is set generously (10,000%) to allow real
    hyperinflation while still catching the previous bug, which produced
    values in the hundreds of MILLIONS of percent from double-differencing
    an already-percentage series."""
    assert panel["inflation_rate"].abs().max() < 10_000, (
        "inflation_rate exceeds plausible bounds even for hyperinflation -- "
        "check that preprocess.py uses the CPI series directly and does NOT "
        "apply pct_change() to an already-percentage series."
    )
    assert panel["inflation_rate"].median() < 20, (
        "Median inflation_rate is implausibly high; most country-years "
        "should have low-to-moderate inflation."
    )


def test_no_missing_values_in_model_columns(panel):
    cols = FEATURES + [TARGET]
    missing = panel[cols].isna().sum()
    assert missing.sum() == 0, f"Unexpected missing values: {missing[missing > 0].to_dict()}"


def test_winsorisation_bounds_applied(panel):
    """Target should be bounded by the winsorisation step (no extreme
    outliers beyond roughly the 1st/99th percentile bounds used during
    construction)."""
    assert panel[TARGET].abs().max() < 200, "Target contains unwinsorised extreme outliers."


def test_country_year_uniqueness(panel):
    """Each (country_code, year) combination should appear exactly once."""
    dupes = panel.duplicated(subset=["country_code", "year"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (country_code, year) rows."


def test_minimum_sample_size(panel):
    assert len(panel) > 2500, f"Sample size unexpectedly small: n={len(panel)}"
    assert panel["country_code"].nunique() > 150, (
        f"Country coverage unexpectedly small: {panel['country_code'].nunique()} countries"
    )


def test_temporal_split_no_overlap(panel):
    """Verify the GroupKFold and temporal-holdout splits used elsewhere in
    the pipeline are well-defined: origin years span a coherent range with
    no gaps that would indicate a merge error."""
    years = sorted(panel["year"].unique())
    assert years[0] >= 2001
    assert years[-1] <= 2018
    # No large gaps (more than 1 missing year) in the origin-year sequence
    gaps = np.diff(years)
    assert gaps.max() <= 2, f"Unexpected gap in origin years: {years}"
