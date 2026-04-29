"""
build_dataset.py — join all ingested sources into a single clean monthly
dataset ready for lag analysis.

What this script does:
  1. Loads S&P 500, unemployment, and tax revenue from CSV / SQLite
  2. Aligns everything to a common monthly index
  3. Flags named downturn events
  4. Outputs data/processed/merged_monthly.csv

Usage:
    python vs-sandbox/build_dataset.py
"""

import sqlite3
import pandas as pd
from pathlib import Path


DB_PATH       = Path(__file__).resolve().parents[1] / "data" / "raw" / "market_data.db"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_PATH   = PROCESSED_DIR / "merged_monthly.csv"


# ── load ─────────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run fetch_prepare_pipeline.py first."
        )
    return sqlite3.connect(DB_PATH)


def load_sp500() -> pd.DataFrame:
    """Load S&P 500 monthly close from SQLite."""
    conn = _get_conn()
    df = pd.read_sql("SELECT date, close FROM sp500 ORDER BY date", conn, parse_dates=["date"], index_col="date")
    conn.close()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df[["close"]].sort_index()


def load_unemployment() -> pd.DataFrame:
    """Load FRED UNRATE from SQLite."""
    conn = _get_conn()
    df = pd.read_sql("SELECT date, rate AS unemployment_rate FROM unemployment ORDER BY date", conn, parse_dates=["date"], index_col="date")
    conn.close()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df


def load_tax() -> pd.DataFrame:
    """Load FRED quarterly tax from SQLite, divide by 12, forward-fill to monthly."""
    conn = _get_conn()
    df = pd.read_sql(
        "SELECT date, receipts_bn FROM tax_revenue WHERE source='FRED_quarterly' ORDER BY date",
        conn, parse_dates=["date"], index_col="date",
    )
    conn.close()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
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
