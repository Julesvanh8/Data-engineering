"""
ingest_unemployment.py — fetch monthly U-3 unemployment rate from the
Bureau of Labor Statistics (BLS) public API v2 and store in SQLite.

BLS series: LNS14000000  (seasonally adjusted U-3 unemployment rate)

Usage:
    python ingestion/ingest_unemployment.py

Dependencies:
    pip install requests pandas
    Optional: set BLS_API_KEY=<your_key> in .env for higher rate limits.
    Register free at https://data.bls.gov/registrationEngine/
"""

import os
import json
import requests
import pandas as pd
from db import get_connection, initialise_db


BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
SERIES_ID   = "LNS14000000"
START_YEAR  = 1948
END_YEAR    = 2026   # BLS API returns up to 20 years per call; we loop in chunks


def fetch_unemployment(api_key: str | None = None) -> pd.DataFrame:
    headers = {"Content-type": "application/json"}
    all_rows = []

    # BLS API allows max 20-year range per request
#    for chunk_start in range(START_YEAR, END_YEAR + 1, 20):
#        chunk_end = min(chunk_start + 19, END_YEAR)
    for chunk_start in range(START_YEAR, END_YEAR + 1, 10):
        chunk_end = min(chunk_start + 9, END_YEAR)
        print(f"  Fetching BLS: {chunk_start}–{chunk_end}")
        payload: dict = {
            "seriesid":  [SERIES_ID],
            "startyear": str(chunk_start),
            "endyear":   str(chunk_end),
        }
        if api_key:
            payload["registrationkey"] = api_key

        resp = requests.post(BLS_API_URL, data=json.dumps(payload), headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data["status"] != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS API error: {data.get('message', data)}")

        for series in data["Results"]["series"]:
            for item in series["data"]:
                year  = int(item["year"])
                month = int(item["period"].replace("M", ""))   # 'M01' → 1
                all_rows.append({
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    "rate": float(item["value"]) if item["value"] != '-' else None,
                })

    df = pd.DataFrame(all_rows).dropna(subset=["rate"]).sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    df.index.name = "date"
    return df


def store_unemployment(df: pd.DataFrame) -> None:
    conn = get_connection()
    cur  = conn.cursor()
    rows = [(str(date.date()), row["rate"]) for date, row in df.iterrows()]
    cur.executemany("""
        INSERT OR REPLACE INTO unemployment (date, rate)
        VALUES (?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"Stored {len(rows)} unemployment rows.")


if __name__ == "__main__":
    initialise_db()
    api_key = os.getenv("BLS_API_KEY")   # set in .env; None = unauthenticated (lower rate limit)
    df = fetch_unemployment(api_key=api_key)
    print(df.tail())
    store_unemployment(df)
