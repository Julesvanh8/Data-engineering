"""
db.py — shared SQLite connection and table initialisation.
All ingestion scripts import get_connection() from here.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "market_data.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialise_db() -> None:
    """Create all tables if they do not already exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS sp500 (
            date        TEXT PRIMARY KEY,   -- YYYY-MM-DD (month-end)
            close       REAL NOT NULL,      -- adjusted closing price
            pct_change  REAL                -- month-over-month % change
        );

        CREATE TABLE IF NOT EXISTS unemployment (
            date        TEXT PRIMARY KEY,   -- YYYY-MM-01 (first of month)
            rate        REAL NOT NULL       -- U-3 unemployment rate (%)
        );

        DROP TABLE IF EXISTS tax_revenue;
        CREATE TABLE tax_revenue (
            date        TEXT NOT NULL,      -- YYYY-MM-01 (first of month)
            receipts_bn REAL NOT NULL,      -- individual income tax receipts (USD billions)
            source      TEXT NOT NULL,      -- 'FRED_quarterly' or 'Treasury_monthly'
            PRIMARY KEY (date, source)
        );

        CREATE TABLE IF NOT EXISTS gdp (
            date        TEXT PRIMARY KEY,   -- YYYY-MM-01 (first of quarter, forward-filled)
            gdp_bn      REAL NOT NULL       -- US GDP in current USD billions
        );

        CREATE TABLE IF NOT EXISTS downturn_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,      -- e.g. 'Dot-com crash'
            start_date  TEXT NOT NULL,      -- YYYY-MM-DD
            end_date    TEXT NOT NULL       -- YYYY-MM-DD (trough)
        );
    """)

    # Seed the three named downturns we analyse
    cur.executemany("""
        INSERT OR IGNORE INTO downturn_events (name, start_date, end_date) VALUES (?, ?, ?)
    """, [
        ("Dot-com crash",  "2000-03-01", "2002-10-01"),
        ("GFC",            "2007-10-01", "2009-03-01"),
        ("COVID crash",    "2020-02-01", "2020-03-01"),
    ])

    conn.commit()
    conn.close()
    print(f"Database initialised at {DB_PATH}")


if __name__ == "__main__":
    initialise_db()
