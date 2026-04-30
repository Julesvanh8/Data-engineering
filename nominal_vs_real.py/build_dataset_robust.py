"""
build_dataset_robust.py — build a monthly dataset for the robustness check.

This script uses the Shiller-style GitHub S&P 500 file with extra variables:
real_close and cpi.

Output:
    data/processed/merged_monthly_vs_robust.csv
"""

import pandas as pd
from pathlib import Path


SP500_ROBUST_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "github_sp500_shiller_robust.csv"
FRED_UNRATE_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "fred_unrate.csv"
FRED_TAX_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "fred_w006rc1q027sbea.csv"

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "merged_monthly_vs_robust.csv"


# =========================
# Load data
# =========================

def load_sp500_robust() -> pd.DataFrame:
    """Load Shiller-style S&P 500 data with nominal and real prices."""
    df = pd.read_csv(SP500_ROBUST_CSV, parse_dates=["date"], index_col="date")

    # Normalize to month-start, same as Valerie's pipeline
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()

    columns_to_keep = [
        "close",
        "real_close",
        "cpi",
    ]

    missing_columns = [
        col for col in columns_to_keep
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in robust S&P 500 file: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[columns_to_keep].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    return df


def load_unemployment() -> pd.DataFrame:
    """Load FRED UNRATE from CSV."""
    df = pd.read_csv(FRED_UNRATE_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()

    df = df.rename(columns={"UNRATE": "unemployment_rate"})

    return df[["unemployment_rate"]].sort_index()


def load_tax() -> pd.DataFrame:
    """
    Load FRED W006RC1Q027SBEA from CSV.
    This is quarterly SAAR, divided by 12 and forward-filled to monthly frequency.
    """
    df = pd.read_csv(FRED_TAX_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()

    df = df.rename(columns={"W006RC1Q027SBEA": "receipts_bn"}).sort_index()

    # Same logic as Valerie's pipeline
    df["receipts_bn"] = df["receipts_bn"] / 12
    df = df.resample("MS").ffill()

    return df[["receipts_bn"]]


# =========================
# Build dataset
# =========================

def build_dataset_robust() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading robustness sources...")

    sp500 = load_sp500_robust()
    unemployment = load_unemployment()
    tax = load_tax()

    print(f"  S&P 500 robust: {len(sp500)} rows ({sp500.index.min().date()} → {sp500.index.max().date()})")
    print(f"  Unemployment:   {len(unemployment)} rows ({unemployment.index.min().date()} → {unemployment.index.max().date()})")
    print(f"  Tax:            {len(tax)} rows ({tax.index.min().date()} → {tax.index.max().date()})")

    print("\nJoining sources...")

    df = sp500.copy()
    df = df.join(unemployment, how="left")
    df = df.join(tax, how="left")

    # Drop rows where macro data is missing
    df = df.dropna(subset=["unemployment_rate", "receipts_bn"])

    # Same base variables as Valerie's dataset
    df["pct_change"] = df["close"].pct_change() * 100
    df["pp_change_unrate"] = df["unemployment_rate"].diff()
    df["pct_change_receipts"] = df["receipts_bn"].pct_change() * 100

    # Robustness variable based on inflation-adjusted S&P 500 price
    df["real_pct_change"] = df["real_close"].pct_change() * 100

    # Keep column order close to Valerie's dataset
    ordered_columns = [
        "close",
        "unemployment_rate",
        "receipts_bn",
        "pct_change",
        "pp_change_unrate",
        "pct_change_receipts",
        "real_close",
        "real_pct_change",
        "cpi",
    ]

    df = df[ordered_columns]

    df.index.name = "date"
    df.to_csv(OUTPUT_PATH)

    print(f"\nSaved robust dataset to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Period: {df.index.min().date()} → {df.index.max().date()}")
    print("Columns:", list(df.columns))

    print("\nPreview:")
    print(df.head())
    print(df.tail())

    return df


if __name__ == "__main__":
    build_dataset_robust()