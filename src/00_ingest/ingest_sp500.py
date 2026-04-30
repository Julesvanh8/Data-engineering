"""
Ingest S&P 500 data from GitHub datasets repository into SQLite database.
Daily data from 1871 to present.
"""
from pathlib import Path
import pandas as pd
import requests
import sqlite3
from typing import Optional


SP500_DATA_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"


def fetch_sp500_daily() -> pd.DataFrame:
    """
    Download S&P 500 data from GitHub datasets repository.
    Returns daily data from 1871 onwards.
    """
    print("Fetching S&P 500 data from GitHub...")
    response = requests.get(SP500_DATA_URL, timeout=60)
    response.raise_for_status()
    
    # Read CSV directly from response
    import io
    df = pd.read_csv(io.StringIO(response.text))
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Rename column to price
    df = df.rename(columns={'SP500': 'price'})
    
    # Convert to numeric and drop NaN
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    
    # Remove duplicate dates
    df = df[~df.index.duplicated(keep='first')]
    
    # Create compatible structure
    result = pd.DataFrame({
        'close': df['price'],
        'adjusted_close': df['price'],
        'volume': pd.NA
    })
    result.index.name = 'date'
    
    print(f"  Fetched {len(result)} daily observations from {result.index.min().date()}")
    return result


def store_sp500_to_sqlite(
    sp500_daily: pd.DataFrame,
    db_path: Optional[Path] = None,
) -> None:
    """
    Store S&P 500 daily data to SQLite database.
    Resamples to monthly and calculates percentage change.
    """
    if db_path is None:
        project_root = Path(__file__).resolve().parents[2]
        db_path = project_root / "data" / "raw" / "market_data.db"
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sp500 (
            date       TEXT PRIMARY KEY,
            close      REAL NOT NULL,
            pct_change REAL
        )
    """)
    
    # Resample daily → monthly (month start), compute pct_change
    monthly = sp500_daily["adjusted_close"].resample("MS").last().dropna()
    pct_ch = monthly.pct_change() * 100
    
    sp500_rows = [
        (str(d.date()), float(c), float(p) if pd.notna(p) else None)
        for d, c, p in zip(monthly.index, monthly, pct_ch)
    ]
    
    cur.executemany(
        "INSERT OR REPLACE INTO sp500 (date, close, pct_change) VALUES (?, ?, ?)",
        sp500_rows,
    )
    
    conn.commit()
    conn.close()
    
    print(f"  ✅ Stored {len(sp500_rows)} monthly S&P 500 records to {db_path.name}")


def main():
    """Main ingest function for S&P 500 data."""
    sp500_daily = fetch_sp500_daily()
    
    # Save raw daily data as CSV backup
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sp500_daily.to_csv(raw_dir / "github_sp500_daily.csv")
    print(f"  Saved raw CSV to {raw_dir / 'github_sp500_daily.csv'}")
    
    # Store to SQLite
    store_sp500_to_sqlite(sp500_daily)


if __name__ == "__main__":
    main()
