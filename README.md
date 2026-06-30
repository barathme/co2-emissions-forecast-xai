# Explainable Machine Learning for Five-Year National CO₂ Emissions Forecasting from Socioeconomic Indicators

This repository contains the complete data processing pipeline, model training code, statistical analysis, figures, and tables supporting the manuscript:

> **Explainable Machine Learning for Five-Year National CO₂ Emissions Forecasting from Socioeconomic Indicators**
> Padma Priya R, Baradeswaran A
> Gandhi Engineering College, Bhubaneswar, Odisha, India

## Overview

This study predicts the **5-year forward percentage change in national CO₂ emissions per capita** from seven lagged socioeconomic indicators (GDP per capita, Human Development Index, internet penetration, inflation, population growth, GDP growth, and relative CO₂ intensity), using a panel of 181–183 countries from 2001–2018 (n=2,986 observations).

Six machine learning models (Linear/Ridge regression, Random Forest, XGBoost, LightGBM, CatBoost) are evaluated against four baselines (naive, country fixed-effects only, country fixed-effects with features) under three independent validation strategies:

- **GroupKFold(5) by country** — eliminates within-country temporal leakage
- **Temporal holdout** (train 2001–2013, test 2014–2018) — assesses forward-time generalisation
- **Leave-one-region-out (LORO)** — assesses cross-regional transferability

All feature attribution (SHAP, permutation importance, partial dependence, individual conditional expectation) is computed consistently from a single XGBoost model.

## Key Results

| Metric | Value |
|---|---|
| XGBoost GroupKFold R² | 0.213 ± 0.054 |
| XGBoost temporal holdout R² | 0.245 (95% CI [0.190, 0.292]) |
| CatBoost temporal holdout R² | 0.246 (95% CI [0.184, 0.301]) |
| Best baseline (country FE + features) | 0.154 (temporal) |
| Top predictors (SHAP) | log GDP per capita, HDI, internet penetration |
| Diebold-Mariano (XGBoost vs. linear) | Significant, p < 0.001 |

Full results are documented in `tables/` and `figures/`, and discussed in the manuscript (`manuscript/`).

## Repository Structure

```
.
├── data/
│   ├── raw/              Original downloaded datasets (or download instructions)
│   ├── processed/         Standardised feature arrays, trained model objects
│   └── cleaned/           Final analysis-ready panel datasets (CSV)
├── src/
│   ├── data/               Data download and feature engineering modules
│   ├── models/             Model training, validation, and statistical testing modules
│   └── visualization/      Figure generation modules
├── scripts/
│   └── run_full_pipeline.py   End-to-end reproduction script
├── notebooks/             Exploratory analysis notebooks (optional, illustrative)
├── tables/                 All manuscript tables as CSV
├── figures/                 All manuscript figures (PNG, 300 DPI)
├── results/                 Saved model metrics, bootstrap outputs, statistical test results
├── tests/                   Unit tests for data integrity and leakage verification
├── manuscript/               Manuscript source files
├── requirements.txt
├── environment.yml
├── Dockerfile
├── CHANGELOG.md
└── LICENSE
```

## Reproducing the Results

### Option 1: Docker (recommended for exact reproducibility)

```bash
docker build -t co2-forecast .
docker run -it co2-forecast python scripts/run_full_pipeline.py
```

### Option 2: Conda environment

```bash
conda env create -f environment.yml
conda activate co2-forecast
python scripts/run_full_pipeline.py
```

### Option 3: pip

```bash
pip install -r requirements.txt
python scripts/run_full_pipeline.py
```

The full pipeline takes approximately 20–30 minutes on a standard laptop CPU (Optuna hyperparameter search is the most time-consuming step; reduce `N_TRIALS` in `src/models/train.py` for a faster run).

## Data Sources

All data are freely available without authentication:

| Dataset | Source | URL |
|---|---|---|
| CO₂ and energy | Our World in Data | github.com/owid/co2-data |
| GDP, Population, CPI | World Bank (via Datahub.io mirror) | github.com/datasets/ |
| Internet penetration, HDI | Gapminder / ITU / UNDP | github.com/open-numbers/ddf--gapminder--systema_globalis |

See `data/raw/README.md` for download instructions and checksums.

## Citing This Work

If you use this code or data pipeline, please cite the manuscript (full citation to be added upon publication) and this repository:

```bibtex
@misc{co2forecast2026,
  title  = {Explainable Machine Learning for Five-Year National CO2 Emissions Forecasting from Socioeconomic Indicators},
  author = {Padma Priya, R. and Baradeswaran, A.},
  year   = {2026},
  note   = {GitHub repository},
  url    = {https://github.com/[ORG]/[REPO]}
}
```

## License

Code: MIT License (see `LICENSE`). Data: subject to the original source licenses (OWID — Creative Commons BY; World Bank — CC BY-4.0; Gapminder — CC BY-4.0).

## Contact

Corresponding author: Baradeswaran A — barathme@yahoo.co.in

## Acknowledgements

This work uses entirely open-access data from Our World in Data, the World Bank, Gapminder, ITU, and UNDP. We thank these organisations for maintaining freely accessible global development data.
