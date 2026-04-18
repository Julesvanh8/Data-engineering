"""
run_all.py — full pipeline: ingest → build dataset → analyse → visualise.

Usage:
    python vs-sandbox/run_all.py
"""
from pathlib import Path
from ingest_tax_fred import load_env
load_env(Path(__file__).resolve().parents[1])  # must be before os.getenv calls

import os
from db import initialise_db
from ingest_sp500 import fetch_sp500, store_sp500
from ingest_unemployment import fetch_unemployment, store_unemployment
from ingest_tax_revenue import fetch_tax_revenue, store_tax_revenue
from ingest_tax_fred import fetch_tax_fred, store_tax_fred
from build_dataset import build_dataset
from analyze_lags import run as run_analysis
from visualize import run as run_visualize

if __name__ == "__main__":
    print("=== Initialising database ===")
    initialise_db()

    print("\n=== S&P 500 ===")
    store_sp500(fetch_sp500())

    print("\n=== Unemployment (BLS) ===")
    store_unemployment(fetch_unemployment(api_key=os.getenv("BLS_API_KEY")))

    print("\n=== Tax revenue (FRED, 1995–2014) ===")
    store_tax_fred(fetch_tax_fred(api_key=os.getenv("FRED_API_KEY", "")))

    print("\n=== Tax revenue (Treasury MTS, 2015–present) ===")
    store_tax_revenue(fetch_tax_revenue())

    print("\n=== Building dataset ===")
    build_dataset()

    print("\n=== Lag analysis ===")
    run_analysis()

    print("\n=== Visualisations ===")
    run_visualize()

    print("\nPipeline complete.")