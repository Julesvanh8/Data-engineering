"""
build_dataset_robust.py — build a monthly dataset for the robustness check.

This script uses the copied SQLite database created by ingest_sp500_robust.py.
It loads unemployment and tax revenue from the same tables as the main dataset,
but replaces the normal S&P 500 input with sp500_robust, which contains
both nominal and inflation-adjusted S&P 500 prices.

Output:
    data/processed/merged_monthly_robust.csv
"""

import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "data" / "raw" / "market_data_robust.db"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "merged_monthly_robust.csv"


def get_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Robust database not found at {DB_PATH}. "
            "Run ingest_sp500_robust.py first."
        )

    return sqlite3.connect(DB_PATH)


def load_sp500_robust() -> pd.DataFrame:
    """Load robust S&P 500 data from SQLite."""
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT date, close, real_close, cpi
        FROM sp500_robust
        ORDER BY date
        """,
        conn,
        parse_dates=["date"],
        index_col="date",
    )

    conn.close()

    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()

    return df[["close", "real_close", "cpi"]].sort_index()


def load_unemployment() -> pd.DataFrame:
    """Load unemployment data from SQLite."""
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT date, rate AS unemployment_rate
        FROM unemployment
        ORDER BY date
        """,
        conn,
        parse_dates=["date"],
        index_col="date",
    )

    conn.close()

    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()

    return df[["unemployment_rate"]].sort_index()


def load_tax() -> pd.DataFrame:
    """
    Load FRED quarterly tax revenue from SQLite.
    The values are divided by 12 and forward-filled to monthly frequency,
    consistent with the main pipeline.
    """
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT date, receipts_bn
        FROM tax_revenue
        WHERE source = 'FRED_quarterly'
        ORDER BY date
        """,
        conn,
        parse_dates=["date"],
        index_col="date",
    )

    conn.close()

    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()

    df["receipts_bn"] = df["receipts_bn"] / 12
    df = df.resample("MS").ffill()

    return df[["receipts_bn"]]


def build_dataset_robust() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading robustness sources from copied SQLite database...")

    sp500 = load_sp500_robust()
    unemployment = load_unemployment()
    tax = load_tax()

    print(
        f"  S&P 500 robust: {len(sp500)} rows "
        f"({sp500.index.min().date()} → {sp500.index.max().date()})"
    )

    print(
        f"  Unemployment:   {len(unemployment)} rows "
        f"({unemployment.index.min().date()} → {unemployment.index.max().date()})"
    )

    print(
        f"  Tax:            {len(tax)} rows "
        f"({tax.index.min().date()} → {tax.index.max().date()})"
    )

    print("\nJoining sources...")

    df = sp500.copy()
    df = df.join(unemployment, how="left")
    df = df.join(tax, how="left")

    df = df.dropna(subset=["unemployment_rate", "receipts_bn"])

    df["pct_change"] = df["close"].pct_change() * 100
    df["pp_change_unrate"] = df["unemployment_rate"].diff()
    df["pct_change_receipts"] = df["receipts_bn"].pct_change() * 100

    df["real_pct_change"] = df["real_close"].pct_change() * 100

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