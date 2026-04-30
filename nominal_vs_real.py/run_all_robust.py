"""
run_all_robust.py — full robustness pipeline for the nominal vs real S&P 500 check.

Important:
    Run the main pipeline first, because this robustness script compares against
    the nominal results created by the main analysis.

Required main output:
    data/processed/events_combined.csv

Robustness pipeline order:
  1. Extract Shiller-style S&P 500 data with nominal close, real_close and CPI.
  2. Store the robust S&P 500 data in a copied SQLite database as sp500_robust.
  3. Build merged_monthly_robust.csv.
  4. Run the lag analysis using real_close instead of nominal close.
  5. Compare events_combined.csv with events_combined_robust.csv.

Usage:
    python vs-sandbox/run_all.py
    python nominal_vs_real.py/run_all_robust.py
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBUST_DIR = Path(__file__).resolve().parent

MAIN_EVENTS = PROJECT_ROOT / "data" / "processed" / "events_combined.csv"


def run_step(step_name: str, script_name: str) -> None:
    script_path = ROBUST_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 70)
    print(step_name)
    print("=" * 70)
    print(f"Running: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with exit code {result.returncode}."
        )

    print(f"{step_name} completed successfully.")


def main() -> None:
    print("\nROBUSTNESS PIPELINE: NOMINAL VS REAL S&P 500")
    print("Project root:", PROJECT_ROOT)
    print("Robustness folder:", ROBUST_DIR)

    if not MAIN_EVENTS.exists():
        raise FileNotFoundError(
            f"Required main output not found: {MAIN_EVENTS}\n"
            "Run the main pipeline first with:\n"
            "python vs-sandbox/run_all.py"
        )

    run_step(
        "Step 1: Extract Shiller-style S&P 500 data",
        "Extract_Shiller.py",
    )

    run_step(
        "Step 2: Store robust S&P 500 data in copied SQLite database",
        "ingest_sp500_robust.py",
    )

    run_step(
        "Step 3: Build robustness dataset",
        "build_dataset_robust.py",
    )

    run_step(
        "Step 4: Run robustness lag analysis",
        "analyze_lags_robust.py",
    )

    run_step(
        "Step 5: Compare nominal and real-price results",
        "compare_robustness.py",
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS PIPELINE COMPLETE")
    print("=" * 70)
    print("Generated outputs:")
    print("  data/raw/github_sp500_shiller_robust.csv")
    print("  data/raw/market_data_robust.db, table sp500_robust")
    print("  data/processed/merged_monthly_robust.csv")
    print("  data/processed/events_combined_robust.csv")
    print("  data/processed/robustness_summary.csv")
    print("  data/processed/robustness_timing_summary.csv")


if __name__ == "__main__":
    main()