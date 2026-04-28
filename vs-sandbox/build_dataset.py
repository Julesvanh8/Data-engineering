"""
build_dataset.py — join all ingested sources into a single clean monthly
dataset ready for lag analysis.

What this script does:
  1. Loads S&P 500, unemployment, and tax revenue from CSV / SQLite
  2. Aligns everything to a common monthly index
  3. Flags named downturn events
  4. Outputs data/processed/merged_monthly_vs.csv

Usage:
    python vs-sandbox/build_dataset.py
"""

import pandas as pd
from pathlib import Path


SHILLER_CSV       = Path(__file__).resolve().parents[1] / "data" / "raw" / "github_sp500_daily.csv"
FRED_UNRATE_CSV   = Path(__file__).resolve().parents[1] / "data" / "raw" / "fred_unrate.csv"
FRED_TAX_CSV      = Path(__file__).resolve().parents[1] / "data" / "raw" / "fred_w006rc1q027sbea.csv"
PROCESSED_DIR     = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_PATH       = PROCESSED_DIR / "merged_monthly_vs.csv"


# ── load ─────────────────────────────────────────────────────────────────────

def load_sp500() -> pd.DataFrame:
    """Load S&P 500 from Shiller historical CSV (1871–present)."""
    df = pd.read_csv(SHILLER_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df[["close"]].sort_index()


def load_unemployment() -> pd.DataFrame:
    """Load FRED UNRATE from CSV (1948–present)."""
    df = pd.read_csv(FRED_UNRATE_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df.rename(columns={"UNRATE": "unemployment_rate"})


def load_tax() -> pd.DataFrame:
    """Load FRED W006RC1Q027SBEA from CSV (quarterly SAAR, 1947–present).
    Divides by 12 for monthly equivalent, forward-fills to monthly frequency.
    """
    df = pd.read_csv(FRED_TAX_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    df = df.rename(columns={"W006RC1Q027SBEA": "receipts_bn"}).sort_index()
    df["receipts_bn"] = df["receipts_bn"] / 12
    df = df.resample("MS").ffill()
    return df

# ── main ─────────────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sources...")
    sp500        = load_sp500()
    unemployment = load_unemployment()
    tax          = load_tax()

    print(f"  S&P 500:      {len(sp500)} rows  ({sp500.index.min().date()} → {sp500.index.max().date()})")
    print(f"  Unemployment: {len(unemployment)} rows  ({unemployment.index.min().date()} → {unemployment.index.max().date()})")
    print(f"  Tax (combined): {len(tax)} rows  ({tax.index.min().date()} → {tax.index.max().date()})")

    print("\nJoining sources...")
    df = sp500.copy()
    df = df.join(unemployment,              how="left")
    df = df.join(tax[["receipts_bn"]],      how="left")

    # Drop rows where we have no data at all for the economic series
    df = df.dropna(subset=["unemployment_rate", "receipts_bn"])

    # Derived columns — computed after join so all series are on the same index
    df["pct_change"]          = df["close"].pct_change() * 100
    df["pp_change_unrate"]    = df["unemployment_rate"].diff()
    df["pct_change_receipts"] = df["receipts_bn"].pct_change() * 100

    df.index.name = "date"
    df.to_csv(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    build_dataset()
