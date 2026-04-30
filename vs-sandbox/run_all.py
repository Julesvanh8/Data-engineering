"""
run_all.py — full pipeline: ingest → build dataset → analyse → visualise + dashboard

Usage:
    python vs-sandbox/run_all.py
"""
from pathlib import Path
from fetch_prepare_pipeline import run_pipeline, load_env_file

import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
from db import initialise_db
from build_dataset import build_dataset
from analyze_lags import run as run_analysis
from visualize import run as run_visualize

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    load_env_file(PROJECT_ROOT)

    print("=== Initialising database ===")
    initialise_db()

    print("\n=== Fetching data & writing to SQLite ===")
    try:
        run_pipeline(
            project_root=PROJECT_ROOT,
            unrate_series="UNRATE",
            tax_series="W006RC1Q027SBEA",
            downturn_threshold=-0.05,
            wb_indicator=None,
        )
    except Exception as e:
        print(f"  WARNING: ingest failed ({e}). Existing DB data will be used.")

    print("\n=== Building dataset ===")
    try:
        root = tk.Tk()
        root.withdraw()
        use_spark = messagebox.askyesno(
            "Pipeline configuration",
            "Use Apache Spark for the ETL build step?\n\n"
            "Spark has a JVM startup and is less efficient for this dataset size.\n\n"
            "Yes = PySpark  (outputs merged_monthly.csv)\n"
            "No  = Pandas   (outputs merged_monthly.csv, faster)"
        )
        root.destroy()
        if use_spark:
            from build_dataset_spark import build_dataset as build_spark
            build_spark()
        else:
            build_dataset()
            analysis_path = None
    except Exception as e:
        print(f"  WARNING: build dataset failed ({e}). Skipping lag analysis and visualisations.")
        analysis_path = None

    print("\n=== Lag analysis ===")
    try:
        run_analysis(data_path=analysis_path)
    except Exception as e:
        print(f"  WARNING: lag analysis failed ({e}).")

    print("\n=== Visualisations ===")
    try:
        run_visualize()
    except Exception as e:
        print(f"  WARNING: visualisations failed ({e}).")

    print("\nPipeline complete.")

    print("\n=== Launching dashboard ===")
    dashboard = Path(__file__).resolve().parent / "dashboard.py"
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(dashboard)])
    print("  Dashboard running in background — open http://localhost:8501")
