#!/usr/bin/env python3
"""
Master pipeline orchestrator for the data engineering project.

This script runs the complete pipeline in the correct order:
  1. Ingest: Fetch raw data and store in SQLite
  2. Transform: Run DBT models to create analytics-ready tables
  3. Analyze: Generate event analysis and visualizations
  4. Dashboard: Instructions to launch interactive dashboard

Usage:
    python src/orchestration/run_pipeline.py [--skip-ingest] [--skip-transform] [--skip-analysis]
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGEST_SCRIPT = PROJECT_ROOT / "src" / "00_ingest" / "run_ingest.py"
DBT_PROJECT = PROJECT_ROOT / "src" / "01_transform" / "transform"
ANALYZE_SCRIPT = PROJECT_ROOT / "src" / "02_analysis" / "analyze_lags.py"
VISUALIZE_SCRIPT = PROJECT_ROOT / "src" / "02_analysis" / "visualize.py"
DASHBOARD_SCRIPT = PROJECT_ROOT / "src" / "03_dashboard" / "dashboard.py"


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 70
    print("\n" + char * width)
    print(f"{title:^{width}}")
    print(char * width + "\n")


def run_command(cmd: list[str], description: str, cwd: Path = None) -> bool:
    """
    Run a shell command and return success status.
    
    Args:
        cmd: Command as list of strings
        description: Description to show user
        cwd: Working directory (optional)
    
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"▶ {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"  ✅ Success\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed with exit code {e.returncode}\n")
        return False
    except FileNotFoundError:
        print(f"  ❌ Command not found: {cmd[0]}\n")
        return False


def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Run the complete data pipeline")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion")
    parser.add_argument("--skip-transform", action="store_true", help="Skip DBT transformations")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis & visualization")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print_section("DATA ENGINEERING PIPELINE", "=")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    
    # Phase 1: Data Ingestion
    if not args.skip_ingest:
        print_section("PHASE 1: DATA INGESTION", "-")
        success = run_command(
            [sys.executable, str(INGEST_SCRIPT)],
            "Fetching S&P 500, unemployment, and tax revenue data"
        )
        if not success:
            print("❌ Ingestion failed. Pipeline stopped.")
            return 1
    else:
        print_section("PHASE 1: DATA INGESTION (SKIPPED)", "-")
    
    # Phase 2: DBT Transformations
    if not args.skip_transform:
        print_section("PHASE 2: DBT TRANSFORMATIONS", "-")
        success = run_command(
            ["dbt", "run"],
            "Running DBT models (staging → intermediate → marts)",
            cwd=DBT_PROJECT
        )
        if not success:
            print("❌ DBT transformation failed. Pipeline stopped.")
            return 1
        
        # Run DBT tests
        print()
        run_command(
            ["dbt", "test"],
            "Running DBT tests",
            cwd=DBT_PROJECT
        )
    else:
        print_section("PHASE 2: DBT TRANSFORMATIONS (SKIPPED)", "-")
    
    # Phase 3: Analysis
    if not args.skip_analysis:
        print_section("PHASE 3: ANALYSIS & VISUALIZATION", "-")
        
        # Run lag analysis
        success = run_command(
            [sys.executable, str(ANALYZE_SCRIPT)],
            "Running event-study lag analysis"
        )
        if not success:
            print("⚠️  Analysis failed, but continuing...")
        
        # Generate visualizations
        print()
        success = run_command(
            [sys.executable, str(VISUALIZE_SCRIPT)],
            "Generating static visualizations"
        )
        if not success:
            print("⚠️  Visualization failed, but continuing...")
    else:
        print_section("PHASE 3: ANALYSIS & VISUALIZATION (SKIPPED)", "-")
    
    # Phase 4: Dashboard Instructions
    print_section("PHASE 4: INTERACTIVE DASHBOARD", "-")
    print("To launch the interactive Streamlit dashboard, run:")
    print(f"\n  streamlit run {DASHBOARD_SCRIPT.relative_to(PROJECT_ROOT)}\n")
    print("The dashboard will open in your browser at http://localhost:8501")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_section("PIPELINE COMPLETE", "=")
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration.total_seconds():.1f} seconds")
    print()
    print("✅ All pipeline stages completed successfully!")
    print()
    print("Output locations:")
    print(f"  - Raw data:        {PROJECT_ROOT / 'data' / 'raw'}")
    print(f"  - DBT models:      {PROJECT_ROOT / 'data' / 'raw' / 'main_marts.db'}")
    print(f"  - Analysis:        {PROJECT_ROOT / 'data' / 'processed'}")
    print(f"  - Visualizations:  {PROJECT_ROOT / 'outputs' / 'figures'}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
