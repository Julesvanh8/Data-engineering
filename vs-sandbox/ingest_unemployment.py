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

from pathlib import Path
import os
import json
import requests
import pandas as pd
from db import get_connection, initialise_db


BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
SERIES_ID = "LNS14000000"
START_YEAR = 1948
END_YEAR = 2026


def load_env(project_root: Path) -> None:
    env_path = project_root / ".env"

    if not env_path.exists():
        print("Geen .env gevonden op:", env_path)
        return

    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and os.environ.get(key) is None:
            os.environ[key] = value


def fetch_unemployment(api_key: str | None = None) -> pd.DataFrame:
    headers = {"Content-type": "application/json"}
    all_rows = []

    for chunk_start in range(START_YEAR, END_YEAR + 1, 10):
        chunk_end = min(chunk_start + 9, END_YEAR)

        print(f"  Fetching BLS: {chunk_start}–{chunk_end}")

        payload: dict = {
            "seriesid": [SERIES_ID],
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
        }

        if api_key:
            payload["registrationkey"] = api_key

        resp = requests.post(
            BLS_API_URL,
            data=json.dumps(payload),
            headers=headers,
            timeout=30
        )

        resp.raise_for_status()
        data = resp.json()

        if data["status"] != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS API error: {data.get('message', data)}")

        for series in data["Results"]["series"]:
            for item in series["data"]:
                year = int(item["year"])
                if item["period"] == "M13":
                    continue
                month = int(item["period"].replace("M", ""))

                all_rows.append({
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    "rate": float(item["value"]) if item["value"] != "-" else None,
                })

    df = (
        pd.DataFrame(all_rows)
        .dropna(subset=["rate"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    df = df.set_index("date")
    df.index.name = "date"

    return df


def store_unemployment(df: pd.DataFrame) -> None:
    conn = get_connection()
    cur = conn.cursor()

    rows = [
        (str(date.date()), row["rate"])
        for date, row in df.iterrows()
    ]

    cur.executemany("""
        INSERT OR REPLACE INTO unemployment (date, rate)
        VALUES (?, ?)
    """, rows)

    conn.commit()
    conn.close()

    print(f"Stored {len(rows)} unemployment rows.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    load_env(project_root)

    api_key = os.getenv("BLS_API_KEY")

    print("BLS API key loaded:", api_key is not None and len(api_key) > 0)

    initialise_db()

    df = fetch_unemployment(api_key=api_key)

    print(df.tail())

    store_unemployment(df)