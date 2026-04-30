"""
Ingest federal income tax revenue data (W006RC1Q027SBEA) from FRED API into SQLite.
Quarterly SAAR (Seasonally Adjusted Annual Rate) data.
"""
from pathlib import Path
import os
import time
import pandas as pd
import requests
import sqlite3
from typing import Optional


FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


def load_env_file() -> None:
    """Load environment variables from .env file."""
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    
    if not env_path.exists():
        return
    
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        existing = os.environ.get(key)
        if key and (existing is None or existing.strip() == ""):
            os.environ[key] = value


def validate_api_key(api_key: str, key_name: str) -> None:
    """Validate that API key is set."""
    if not api_key or api_key.strip() == "" or api_key.strip() == "Type hier je API key":
        raise ValueError(
            f"{key_name} missing. Set your key in .env file as {key_name}=your_key_here"
        )


def fetch_fred_series(series_id: str, api_key: str) -> pd.Series:
    """
    Fetch a FRED series with retry logic for server errors.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    
    last_status: Optional[int] = None
    for attempt in range(4):
        response = requests.get(FRED_BASE_URL, params=params, timeout=60)
        last_status = response.status_code
        if 500 <= response.status_code < 600:
            time.sleep(1.5 * (attempt + 1))
            continue
        response.raise_for_status()
        payload = response.json()
        break
    else:
        raise ValueError(
            f"FRED temporarily unavailable for {series_id} (last status: {last_status})"
        )
    
    if "observations" not in payload:
        raise ValueError(f"Unexpected FRED response for {series_id}: {payload}")
    
    observations = payload["observations"]
    frame = pd.DataFrame(observations)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    
    series = frame.set_index("date")["value"].sort_index()
    series.name = series_id
    return series


def store_tax_revenue_to_sqlite(
    tax_revenue: pd.Series,
    db_path: Optional[Path] = None,
) -> None:
    """
    Store tax revenue data to SQLite database.
    Stores raw quarterly SAAR values; transformation to monthly happens in DBT.
    """
    if db_path is None:
        project_root = Path(__file__).resolve().parents[2]
        db_path = project_root / "data" / "raw" / "market_data.db"
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tax_revenue (
            date        TEXT    NOT NULL,
            receipts_bn REAL    NOT NULL,
            source      TEXT    NOT NULL,
            PRIMARY KEY (date, source)
        )
    """)
    
    # Prepare rows - store raw quarterly SAAR
    tax_rows = [
        (str(d.date()), float(v), "FRED_quarterly")
        for d, v in tax_revenue.items()
        if pd.notna(v)
    ]
    
    cur.executemany(
        "INSERT OR REPLACE INTO tax_revenue (date, receipts_bn, source) VALUES (?, ?, ?)",
        tax_rows,
    )
    
    conn.commit()
    conn.close()
    
    print(f"  ✅ Stored {len(tax_rows)} tax revenue records to {db_path.name}")


def main():
    """Main ingest function for tax revenue data."""
    load_env_file()
    
    fred_key = os.getenv("FRED_API_KEY", FRED_API_KEY)
    validate_api_key(fred_key, "FRED_API_KEY")
    
    print("Fetching tax revenue data from FRED (W006RC1Q027SBEA)...")
    tax_revenue = fetch_fred_series("W006RC1Q027SBEA", api_key=fred_key)
    
    # Save raw CSV backup
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tax_revenue.to_frame().to_csv(raw_dir / "fred_w006rc1q027sbea.csv")
    print(f"  Saved raw CSV to {raw_dir / 'fred_w006rc1q027sbea.csv'}")
    
    # Store to SQLite
    store_tax_revenue_to_sqlite(tax_revenue)


if __name__ == "__main__":
    main()
