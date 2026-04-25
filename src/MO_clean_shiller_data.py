"""
clean_shiller_data.py

Clean Shiller S&P 500 data and make it compatible with the team pipeline.

What this script does:
1. Loads the Shiller Excel file
2. Cleans the date, price and CPI columns
3. Converts the date to month-start, matching  pipeline
4. Creates close and pct_change columns, matching the yfinance structure
5. Adds real prices, real returns, downturn indicators, lag features and rolling features
6. Saves the cleaned dataset to data/processed/shiller_sp500_monthly_clean.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# 1. Bestandslocaties
# =========================

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path(
        r"C:/Vakken 2de semester 2026-2027/Data Engineering/Data Engineering project/Data-engineering"
    )

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXCEL_FILE = RAW_DIR / "Shiller_data.xls"
OUTPUT_FILE = PROCESSED_DIR / "shiller_sp500_monthly_clean.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DOWNTURN_THRESHOLD = -5
START_DATE = "1948-01-01"
END_DATE = "2025-12-31"


# =========================
# 2. Zoek automatisch de echte header row
# =========================

def find_header_row(excel_file, sheet_name="Data"):
    preview = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        header=None,
        nrows=50
    )

    for index, row in preview.iterrows():
        values = row.astype(str).str.strip().str.lower().tolist()

        if "date" in values and "p" in values and "cpi" in values:
            return index

    raise ValueError("De header row met Date, P en CPI werd niet gevonden.")


# =========================
# 3. Zet Shiller datum om naar month-start
# =========================

def convert_shiller_date(value):
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip().replace(",", ".")

    try:
        number = float(text)
        year = int(np.floor(number))
        month = int(round((number - year) * 100))

        if month < 1 or month > 12:
            return pd.NaT

        return pd.Timestamp(year=year, month=month, day=1)

    except ValueError:
        date = pd.to_datetime(text, errors="coerce")

        if pd.isna(date):
            return pd.NaT

        return date.to_period("M").to_timestamp()


# =========================
# 4. Load en clean Shiller data
# =========================

def load_shiller():
    header_row = find_header_row(EXCEL_FILE, sheet_name="Data")

    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name="Data",
        skiprows=header_row
    )

    df.columns = [
        str(col).strip().replace("\n", " ").replace("\r", " ")
        for col in df.columns
    ]

    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]

    required_columns = ["Date", "P", "CPI"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Deze kolommen ontbreken: {missing_columns}. Beschikbare kolommen zijn: {list(df.columns)}"
        )

    shiller = df[["Date", "P", "CPI"]].copy()

    shiller = shiller.rename(
        columns={
            "Date": "date_raw",
            "P": "close",
            "CPI": "cpi"
        }
    )

    shiller["date"] = shiller["date_raw"].apply(convert_shiller_date)
    shiller["close"] = pd.to_numeric(shiller["close"], errors="coerce")
    shiller["cpi"] = pd.to_numeric(shiller["cpi"], errors="coerce")

    shiller = shiller.dropna(subset=["date", "close", "cpi"])
    shiller = shiller.sort_values("date")
    shiller = shiller.drop_duplicates(subset="date", keep="last")

    return shiller


# =========================
# 5. Maak analysevariabelen
# =========================

def add_features(shiller):
    shiller = shiller.copy()

    base_cpi = shiller["cpi"].iloc[-1]

    shiller["sp500_real_price"] = (
        shiller["close"] / shiller["cpi"] * base_cpi
    )

    shiller["pct_change"] = (
        shiller["close"].pct_change() * 100
    )

    shiller["sp500_return_real_pct"] = (
        shiller["sp500_real_price"].pct_change() * 100
    )

    shiller["downturn"] = (
        shiller["pct_change"] <= DOWNTURN_THRESHOLD
    ).astype("Int64")

    shiller.loc[shiller["pct_change"].isna(), "downturn"] = pd.NA

    shiller["downturn_real"] = (
        shiller["sp500_return_real_pct"] <= DOWNTURN_THRESHOLD
    ).astype("Int64")

    shiller.loc[shiller["sp500_return_real_pct"].isna(), "downturn_real"] = pd.NA

    for lag in range(1, 13):
        shiller[f"downturn_lag_{lag}m"] = shiller["downturn"].shift(lag)
        shiller[f"downturn_real_lag_{lag}m"] = shiller["downturn_real"].shift(lag)

    shiller["return_rolling_mean_3m"] = (
        shiller["pct_change"].rolling(window=3).mean()
    )

    shiller["return_rolling_mean_6m"] = (
        shiller["pct_change"].rolling(window=6).mean()
    )

    shiller["return_rolling_volatility_3m"] = (
        shiller["pct_change"].rolling(window=3).std()
    )

    shiller["return_rolling_volatility_6m"] = (
        shiller["pct_change"].rolling(window=6).std()
    )

    shiller["year"] = shiller["date"].dt.year
    shiller["month"] = shiller["date"].dt.month
    shiller["quarter"] = shiller["date"].dt.quarter

    return shiller


# =========================
# 6. Build dataset
# =========================

def build_shiller_dataset():
    print("Loading Shiller data...")

    shiller = load_shiller()
    shiller = add_features(shiller)

    shiller = shiller[
        (shiller["date"] >= START_DATE) &
        (shiller["date"] <= END_DATE)
    ].copy()

    columns_first = [
        "date",
        "close",
        "pct_change",
        "downturn",
        "cpi",
        "sp500_real_price",
        "sp500_return_real_pct",
        "downturn_real",
        "year",
        "month",
        "quarter"
    ]

    other_columns = [
        col for col in shiller.columns
        if col not in columns_first and col != "date_raw"
    ]

    shiller = shiller[columns_first + other_columns]

    shiller.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned Shiller dataset to: {OUTPUT_FILE}")
    print(f"Rows: {len(shiller)}")
    print(f"Period: {shiller['date'].min().date()} to {shiller['date'].max().date()}")
    print(f"Nominal downturns: {int(shiller['downturn'].sum())}")
    print(f"Real downturns: {int(shiller['downturn_real'].sum())}")

    print(shiller.head())
    print(shiller.tail())

    return shiller


if __name__ == "__main__":
    build_shiller_dataset()