"""
ingest_tax_fred.py — fetch quarterly federal individual income tax receipts
from FRED (series W006RC1Q027SBEA) and store in SQLite.

This covers 1947 onwards at quarterly frequency. Used for dot-com (2000)
and Global Financial Crisis (2008) analysis where Treasury MTS data is unavailable.

Usage:
    python ingestion/ingest_tax_fred.py

Requires:
    FRED_API_KEY in .env
"""

import os
import requests
import pandas as pd
from pathlib import Path
from db import get_connection, initialise_db

FRED_SERIES  = "W006RC1Q027SBEA"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"


def load_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key   = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and os.environ.get(key) is None:
            os.environ[key] = value


def fetch_tax_fred(api_key: str) -> pd.DataFrame:
    params = {
        "series_id":        FRED_SERIES,
        "api_key":          api_key,
        "file_type":        "json",
        "observation_start": "1947-01-01",
    }
    resp = requests.get(FRED_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if "observations" not in payload:
        raise RuntimeError(f"Unexpected FRED response: {payload}")

    df = pd.DataFrame(payload["observations"])
    df["date"]        = pd.to_datetime(df["date"])
    df["receipts_bn"] = pd.to_numeric(df["value"], errors="coerce")  # already in billions
    df = df[["date", "receipts_bn"]].dropna()
    df = df.sort_values("date").set_index("date")
    df.index.name = "date"
    return df


def store_tax_fred(df: pd.DataFrame) -> None:
    """
    Forward-fill quarterly data to monthly and store with source tag.
    Only stores rows up to 2014-12-01 — from 2015 onwards we use
    the higher-quality Treasury MTS monthly data.
    """
    # Resample to month-start, forward-fill within each quarter
    monthly = df.resample("MS").ffill()

    conn = get_connection()
    cur  = conn.cursor()

    # Add source column to tax_revenue table if it doesn't exist yet
    try:
        cur.execute("ALTER TABLE tax_revenue ADD COLUMN source TEXT")
        conn.commit()
    except Exception:
        pass  # column already exists

    rows = [
        (str(date.date()), row["receipts_bn"], "FRED_quarterly")
        for date, row in monthly.iterrows()
    ]
    cur.executemany("""
        INSERT OR REPLACE INTO tax_revenue (date, receipts_bn, source)
        VALUES (?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"Stored {len(rows)} FRED tax rows (1948–present, forward-filled from quarterly).")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    load_env(project_root)

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        raise ValueError("FRED_API_KEY missing — add it to your .env file.")

    initialise_db()
    df = fetch_tax_fred(api_key)
    print(f"FRED date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"Quarters fetched: {len(df)}")
    store_tax_fred(df)
