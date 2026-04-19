"""
ingest_tax_revenue.py — fetch monthly individual income tax receipts from
the US Treasury Fiscal Data API and store in SQLite.

Endpoint: https://fiscaldata.treasury.gov/api/public/debt/v2/accounting/mts/mts_table_4/
We filter for:
  classification_desc = 'Individual Income Taxes'
  record_type_cd      = 'RSG'  (regular, not footnotes)

Usage:
    python ingestion/ingest_tax_revenue.py

Dependencies:
    pip install requests pandas
"""

import requests
import pandas as pd
from db import get_connection, initialise_db


BASE_URL = "https://api.fiscaldata.treasury.gov//services/api/fiscal_service/v1/accounting/mts/mts_table_4"
FIELDS = "record_date,classification_desc,current_month_net_rcpt_amt,record_type_cd"
FILTER = "classification_desc:eq:Total -- Individual Income Taxes,record_type_cd:eq:RSG"
PAGE_SIZE = 1000


def fetch_tax_revenue() -> pd.DataFrame:
    all_rows = []
    page = 1

    while True:
        params = {
            "fields":          FIELDS,
            "filter":          FILTER,
            "page[number]":    page,
            "page[size]":      PAGE_SIZE,
            "sort":            "record_date",
        }
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        data = payload.get("data", [])
        if not data:
            break

        all_rows.extend(data)

        # Stop if we have fetched all pages
        meta        = payload.get("meta", {}).get("pagination", {})
        total_pages = meta.get("total_pages", 1)
        if page >= int(total_pages):
            break
        page += 1

    if not all_rows:
        raise RuntimeError("Treasury API returned no data.")

    df = pd.DataFrame(all_rows)
    df["date"]        = pd.to_datetime(df["record_date"]).dt.to_period("M").dt.to_timestamp()
    df["receipts_bn"] = pd.to_numeric(df["current_month_net_rcpt_amt"], errors="coerce") / 1_000_000_000
    df = df[["date", "receipts_bn"]].dropna()
    df = df[df["date"] >= "1948-01-01"]
    df = df.sort_values("date").set_index("date")
    df.index.name = "date"
    return df

def store_tax_revenue(df: pd.DataFrame) -> None:
    conn = get_connection()
    cur  = conn.cursor()

    try:
        cur.execute("ALTER TABLE tax_revenue ADD COLUMN source TEXT")
        conn.commit()
    except Exception:
        pass  # column already exists

    rows = [
        (str(date.date()), row["receipts_bn"], "Treasury_monthly")
        for date, row in df.iterrows()
    ]
    cur.executemany("""
        INSERT OR REPLACE INTO tax_revenue (date, receipts_bn, source)
        VALUES (?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"Stored {len(rows)} tax revenue rows.")

if __name__ == "__main__":
    initialise_db()
    df = fetch_tax_revenue()
    print(df.tail())
    store_tax_revenue(df)
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"Row count: {len(df)}")