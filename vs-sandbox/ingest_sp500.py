"""
ingest_sp500.py — fetch monthly S&P 500 closing prices via yfinance
and store them in the local SQLite database.

Usage:
    python ingestion/ingest_sp500.py

Dependencies:
    pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd
from db import get_connection, initialise_db


START_DATE = "1995-01-01"   # covers all three downturn events with context
TICKER     = "^GSPC"        # S&P 500


def fetch_sp500() -> pd.DataFrame:
    raw = yf.download(TICKER, start=START_DATE, interval="1mo", auto_adjust=True, progress=False)

    if raw.empty:
        raise RuntimeError("yfinance returned no data — check your internet connection.")

    # Keep only the adjusted close; resample to month-end to normalise the index
    df = raw[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample("ME").last()                        # last trading day of each month
    df.index = df.index.to_period("M").to_timestamp()   # normalise to month-start for joining
    df.columns = ["close"]
    df["pct_change"] = df["close"].pct_change() * 100
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    return df


def store_sp500(df: pd.DataFrame) -> None:
    conn = get_connection()
    cur  = conn.cursor()
    rows = [
        (str(date.date()), row["close"], row["pct_change"] if pd.notna(row["pct_change"]) else None)
        for date, row in df.iterrows()
    ]
    cur.executemany("""
        INSERT OR REPLACE INTO sp500 (date, close, pct_change)
        VALUES (?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"Stored {len(rows)} S&P 500 rows.")


if __name__ == "__main__":
    initialise_db()
    df = fetch_sp500()
    print(df.tail())
    store_sp500(df)
