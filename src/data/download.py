"""
download.py
============
Downloads the six open-access datasets used in this study and writes a
checksum/provenance report for reproducibility auditing.

Usage:
    python download.py --output-dir data/raw
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import requests

DATASETS = {
    "owid_co2_macro.csv": (
        "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    ),
    "gdp.csv": (
        "https://raw.githubusercontent.com/datasets/gdp/main/data/gdp.csv"
    ),
    "population.csv": (
        "https://raw.githubusercontent.com/datasets/population/main/data/population.csv"
    ),
    "cpi_inflation.csv": (
        "https://raw.githubusercontent.com/datasets/imf-weo/main/data/"
        "weo-cpi-inflation.csv"
    ),
    "gapminder_internet.csv": (
        "https://raw.githubusercontent.com/open-numbers/"
        "ddf--gapminder--systema_globalis/master/ddf--datapoints--"
        "internet_users_pct--by--geo--time.csv"
    ),
    "gapminder_hdi.csv": (
        "https://raw.githubusercontent.com/open-numbers/"
        "ddf--gapminder--systema_globalis/master/ddf--datapoints--"
        "hdi_human_development_index--by--geo--time.csv"
    ),
}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as exc:
        print(f"  FAILED: {url} -> {exc}", file=sys.stderr)
        return False


def main(output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for filename, url in DATASETS.items():
        dest = out / filename
        print(f"Downloading {filename} ...")
        ok = download_file(url, dest)
        if not ok:
            print(
                f"  Skipped (download failed). If this persists, fetch manually "
                f"from: {url}",
                file=sys.stderr,
            )
            continue
        try:
            df = pd.read_csv(dest, low_memory=False)
            n_rows, n_cols = df.shape
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not parse {filename} as CSV: {exc}")
            n_rows, n_cols = None, None

        checksum = sha256_of_file(dest)
        size_kb = dest.stat().st_size / 1024
        report_rows.append(
            {
                "filename": filename,
                "url": url,
                "rows": n_rows,
                "columns": n_cols,
                "size_kb": round(size_kb, 1),
                "sha256": checksum,
            }
        )
        print(f"  OK: {n_rows} rows x {n_cols} cols, {size_kb:.1f} KB, sha256={checksum[:12]}...")

    report = pd.DataFrame(report_rows)
    report_path = out / "download_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\nProvenance report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save downloaded files (default: data/raw)",
    )
    args = parser.parse_args()
    main(args.output_dir)
