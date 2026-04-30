"""
ingest_sp500_robust.py — load Shiller-style S&P 500 robustness data
and store it in a copy of the SQLite database.

Input:
    data/raw/github_sp500_shiller_robust.csv
    data/raw/market_data.db

Output database:
    data/raw/market_data_robust.db

Output table:
    sp500_robust

This script does not modify the original market_data.db.
It first copies market_data.db to market_data_robust.db and then adds
the sp500_robust table to the copied database.
"""

import shutil
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

MAIN_DB_PATH = RAW_DIR / "market_data.db"
ROBUST_DB_PATH = RAW_DIR / "market_data_robust.db"

SP500_ROBUST_CSV = RAW_DIR / "github_sp500_shiller_robust.csv"


def create_robust_database_copy() -> None:
    """
    Create a fresh copy of market_data.db for the robustness check.
    The original database remains unchanged.
    """
    if not MAIN_DB_PATH.exists():
        raise FileNotFoundError(
            f"Main database not found at {MAIN_DB_PATH}. "
            "Run the main ingestion pipeline first."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ROBUST_DB_PATH.exists():
        ROBUST_DB_PATH.unlink()

    shutil.copy2(MAIN_DB_PATH, ROBUST_DB_PATH)

    print(f"Copied main database to: {ROBUST_DB_PATH}")


def get_connection() -> sqlite3.Connection:
    if not ROBUST_DB_PATH.exists():
        raise FileNotFoundError(
            f"Robust database not found at {ROBUST_DB_PATH}. "
            "Create the robust database copy first."
        )

    return sqlite3.connect(ROBUST_DB_PATH)


def initialise_sp500_robust_table() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sp500_robust (
            date TEXT PRIMARY KEY,
            close REAL NOT NULL,
            real_close REAL NOT NULL,
            cpi REAL NOT NULL
        );
    """)

    conn.commit()
    conn.close()


def load_sp500_robust_csv() -> pd.DataFrame:
    if not SP500_ROBUST_CSV.exists():
        raise FileNotFoundError(
            f"Robust S&P 500 file not found at {SP500_ROBUST_CSV}. "
            "Run Extract_Shiller.py first."
        )

    df = pd.read_csv(SP500_ROBUST_CSV, parse_dates=["date"])

    required_columns = ["date", "close", "real_close", "cpi"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in robust S&P 500 file: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[required_columns].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["real_close"] = pd.to_numeric(df["real_close"], errors="coerce")
    df["cpi"] = pd.to_numeric(df["cpi"], errors="coerce")

    df = df.dropna(subset=["date", "close", "real_close", "cpi"])

    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("date")
    df = df.drop_duplicates(subset="date", keep="last")

    return df


def store_sp500_robust(df: pd.DataFrame) -> None:
    initialise_sp500_robust_table()

    conn = get_connection()
    cur = conn.cursor()

    rows = [
        (
            str(row["date"].date()),
            float(row["close"]),
            float(row["real_close"]),
            float(row["cpi"]),
        )
        for _, row in df.iterrows()
    ]

    cur.executemany("""
        INSERT OR REPLACE INTO sp500_robust (
            date,
            close,
            real_close,
            cpi
        )
        VALUES (?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()

    print(f"Stored {len(rows)} rows in table sp500_robust.")


def main() -> None:
    print("Creating robust database copy...")
    create_robust_database_copy()

    print("\nLoading robust S&P 500 data...")
    df = load_sp500_robust_csv()

    print(f"Rows loaded: {len(df)}")
    print(f"Period: {df['date'].min().date()} → {df['date'].max().date()}")
    print("Columns:", df.columns.tolist())

    print("\nWriting robust S&P 500 data to copied SQLite database...")
    store_sp500_robust(df)

    print("\nDone.")
    print(f"Original database unchanged: {MAIN_DB_PATH}")
    print(f"Robust database created:     {ROBUST_DB_PATH}")


if __name__ == "__main__":
    main()