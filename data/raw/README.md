# Raw Data

This directory holds the original, unmodified datasets used in this study. Raw CSV files are not committed to this repository (to keep it lightweight); instead, use `src/data/download.py` to fetch them, or follow the manual instructions below. All sources are freely accessible without authentication.

## Datasets

| File | Source | URL | Notes |
|---|---|---|---|
| `owid_co2_macro.csv` | Our World in Data | https://github.com/owid/co2-data | CO₂ emissions, energy, GDP, population, 218 countries, 2000–2023 |
| `gdp.csv` | World Bank (via Datahub.io) | https://github.com/datasets/gdp | GDP, current USD, 266 countries, 2000–2023 |
| `population.csv` | World Bank (via Datahub.io) | https://github.com/datasets/population | Total population, 266 countries, 2000–2023 |
| `cpi_inflation.csv` | World Bank (via Datahub.io) | https://github.com/datasets/cpi | CPI inflation, annual %, 193 countries, 2000–2023 |
| `gapminder_internet.csv` | Gapminder / ITU | https://github.com/open-numbers/ddf--gapminder--systema_globalis | Internet users (% of population), 193 countries, 2000–2022 |
| `gapminder_hdi.csv` | Gapminder / UNDP | https://github.com/open-numbers/ddf--gapminder--systema_globalis | Human Development Index, 195 countries, 2000–2022 |
| `gapminder_geo.csv` | Gapminder | same as above | Country-to-region mapping used for LORO validation |

All files were accessed 26 June 2025. SHA-256 checksums of the exact files used in this study are recorded in `download_report.csv` (generated automatically by the download script).

## Downloading

```bash
python ../../src/data/download.py --output-dir .
```

This will fetch all six datasets, verify they are well-formed, and write `download_report.csv` with file sizes, row counts, and SHA-256 checksums for reproducibility auditing.

## Manual Download (if the script is unavailable)

Each dataset can also be downloaded directly from its GitHub repository's raw CSV link. See the URLs above. No API key, registration, or authentication is required for any source.

## License and Attribution

- **Our World in Data**: Creative Commons BY 4.0. Cite: Ritchie H, Rosado P, Roser M. CO₂ and Greenhouse Gas Emissions. Our World in Data; 2024.
- **World Bank**: Creative Commons Attribution 4.0 (CC BY-4.0).
- **Gapminder**: Creative Commons Attribution 4.0 (CC BY-4.0).
- **ITU / UNDP** (via Gapminder mirror): subject to original ITU/UNDP terms; redistributed by Gapminder under CC BY-4.0.

Please retain attribution to the original data providers in any derivative work.
