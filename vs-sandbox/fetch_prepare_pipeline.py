from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


FRED_API_KEY = os.getenv("FRED_API_KEY", "")

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2/country/USA/indicator"
SP500_DATA_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"


def load_env_file(project_root: Path) -> None:
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
    if not api_key or api_key.strip() == "" or api_key.strip() == "Type hier je API key":
        raise ValueError(
            f"{key_name} ontbreekt. Zet je sleutel in een .env-bestand als {key_name}=Type hier je API key (vervang daarna de placeholder)."
        )


def fetch_sp500_github() -> pd.DataFrame:
    """
    Download S&P 500 data van de GitHub datasets repository.
    Deze dataset bevat historische data vanaf 1871.
    """
    response = requests.get(SP500_DATA_URL, timeout=60)
    response.raise_for_status()
    
    # Lees de CSV direct van de response
    import io
    df = pd.read_csv(io.StringIO(response.text))
    
    # De dataset heeft kolommen: Date, SP500
    # Converteer datum naar datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Hernoem de kolom naar price
    df = df.rename(columns={'SP500': 'price'})
    
    # Converteer naar numeriek
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    
    # Verwijder eventuele duplicate dates
    df = df[~df.index.duplicated(keep='first')]
    
    # Maak het compatible met de verwachte structuur
    result = pd.DataFrame({
        'close': df['price'],
        'adjusted_close': df['price'],
        'volume': pd.NA
    })
    result.index.name = 'date'
    
    return result


def fetch_fred_series(series_id: str, api_key: str) -> pd.Series:
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
        raise ValueError(f"FRED tijdelijk niet beschikbaar voor {series_id} (laatste status: {last_status})")

    if "observations" not in payload:
        raise ValueError(f"Onverwachte FRED response voor {series_id}: {payload}")

    observations = payload["observations"]
    frame = pd.DataFrame(observations)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    series = frame.set_index("date")["value"].sort_index()
    series.name = series_id
    return series


def fetch_fred_stock_proxy(series_id: str, api_key: str) -> pd.DataFrame:
    series = fetch_fred_series(series_id=series_id, api_key=api_key)
    frame = series.to_frame(name="adjusted_close")
    frame["close"] = frame["adjusted_close"]
    frame["volume"] = pd.NA
    frame = frame.loc[:, ["close", "adjusted_close", "volume"]]
    frame.index.name = "date"
    return frame


def fetch_stock_daily() -> pd.DataFrame:
    """
    Haal S&P 500 data op via de GitHub datasets repository (dagelijkse data vanaf 1871).
    """
    print("Data ophalen van S&P 500 GitHub dataset...")
    stock_data = fetch_sp500_github()
    print(f"S&P 500 data opgehaald: {len(stock_data)} observaties vanaf {stock_data.index.min().date()}")
    return stock_data


def fetch_world_bank_indicator(indicator: str) -> pd.Series:
    url = f"{WORLD_BANK_BASE_URL}/{indicator}"
    params = {
        "format": "json",
        "per_page": 20000,
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Onverwachte World Bank response voor {indicator}: {payload}")

    data = payload[1]
    frame = pd.DataFrame(data)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y", errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    series = frame.dropna(subset=["date"]).set_index("date")["value"].sort_index()
    series.name = indicator
    return series


def create_monthly_stock_features(daily_prices: pd.DataFrame, downturn_threshold: float) -> pd.DataFrame:
    monthly_close = daily_prices["adjusted_close"].resample("ME").last()
    monthly_return = monthly_close.pct_change()
    downturn_flag = (monthly_return <= downturn_threshold).astype(int)

    stock_features = pd.DataFrame(
        {
            "sp500_proxy_close": monthly_close,
            "sp500_proxy_return": monthly_return,
            "downturn": downturn_flag,
        }
    )
    return stock_features


def harmonize_to_monthly(unrate: pd.Series, tax_revenue: pd.Series) -> pd.DataFrame:
    unrate_monthly = unrate.resample("ME").last().rename("unemployment_rate")

    # Kwartaalreeks omzetten naar maandfrequentie door binnen het kwartaal te forward-fillen.
    tax_monthly = tax_revenue.resample("ME").ffill().rename("federal_income_tax_revenue")

    return pd.concat([unrate_monthly, tax_monthly], axis=1, sort=False)


def add_lag_features(frame: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    output = frame.copy()
    for lag in range(1, max_lag + 1):
        output[f"downturn_lag_{lag}m"] = output["downturn"].shift(lag)
    return output


def ensure_directories(base_path: Path) -> tuple[Path, Path]:
    raw_dir = base_path / "data" / "raw"
    processed_dir = base_path / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir


def store_to_sqlite(
    project_root: Path,
    sp500_daily: pd.DataFrame,
    unrate: pd.Series,
    tax_revenue: pd.Series,
) -> None:
    import sqlite3

    db_path = project_root / "data" / "raw" / "market_data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS sp500 (
            date       TEXT PRIMARY KEY,
            close      REAL NOT NULL,
            pct_change REAL
        );
        CREATE TABLE IF NOT EXISTS unemployment (
            date TEXT PRIMARY KEY,
            rate REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tax_revenue (
            date        TEXT    NOT NULL,
            receipts_bn REAL    NOT NULL,
            source      TEXT    NOT NULL,
            PRIMARY KEY (date, source)
        );
    """)

    # S&P 500 — resample daily → monthly, compute pct_change
    monthly = sp500_daily["adjusted_close"].resample("MS").last().dropna()
    pct_ch  = monthly.pct_change() * 100
    sp500_rows = [
        (str(d.date()), float(c), float(p) if pd.notna(p) else None)
        for d, c, p in zip(monthly.index, monthly, pct_ch)
    ]
    cur.executemany(
        "INSERT OR REPLACE INTO sp500 (date, close, pct_change) VALUES (?, ?, ?)",
        sp500_rows,
    )

    # Unemployment — store as-is (monthly)
    unemp_rows = [
        (str(d.date()), float(v))
        for d, v in unrate.items()
        if pd.notna(v)
    ]
    cur.executemany(
        "INSERT OR REPLACE INTO unemployment (date, rate) VALUES (?, ?)",
        unemp_rows,
    )

    # Tax revenue — store raw quarterly SAAR; build_dataset divides by 12
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
    print(f"  SQLite: {len(sp500_rows)} sp500, {len(unemp_rows)} unemployment, {len(tax_rows)} tax rows → {db_path.name}")


def run_pipeline(
    project_root: Path,
    unrate_series: str,
    tax_series: str,
    downturn_threshold: float,
    wb_indicator: Optional[str],
) -> Path:
    raw_dir, processed_dir = ensure_directories(project_root)

    fred_key = os.getenv("FRED_API_KEY", FRED_API_KEY)
    validate_api_key(fred_key, "FRED_API_KEY")

    # Haal S&P 500 data op via GitHub dataset
    stock_daily = fetch_stock_daily()
    stock_daily.to_csv(raw_dir / "github_sp500_daily.csv")

    unrate = fetch_fred_series(unrate_series, api_key=fred_key)
    unrate.to_frame().to_csv(raw_dir / f"fred_{unrate_series.lower()}.csv")

    tax_revenue = fetch_fred_series(tax_series, api_key=fred_key)
    tax_revenue.to_frame().to_csv(raw_dir / f"fred_{tax_series.lower()}.csv")

    store_to_sqlite(project_root, stock_daily, unrate, tax_revenue)

    stock_monthly = create_monthly_stock_features(stock_daily, downturn_threshold=downturn_threshold)
    macro_monthly = harmonize_to_monthly(unrate=unrate, tax_revenue=tax_revenue)
    macro_monthly = macro_monthly.reindex(macro_monthly.index.union(stock_monthly.index)).sort_index().ffill()
    macro_monthly = macro_monthly.reindex(stock_monthly.index)

    merged = pd.concat([stock_monthly, macro_monthly], axis=1, sort=False).dropna(how="any")
    merged = add_lag_features(merged, max_lag=12)
    merged["stock_data_source"] = "github_sp500"

    if wb_indicator:
        wb_series = fetch_world_bank_indicator(wb_indicator)
        wb_monthly = wb_series.resample("ME").ffill().rename(f"world_bank_{wb_indicator.lower()}")
        wb_monthly.to_frame().to_csv(raw_dir / f"world_bank_{wb_indicator.lower()}.csv")
        merged = merged.join(wb_monthly, how="left")

    merged.index.name = "month"
    output_path = processed_dir / "merged_monthly.csv"
    merged.to_csv(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Haal data op uit GitHub S&P 500 dataset en FRED, definieer downturns en maak een geharmoniseerde maanddataset."
        )
    )
    parser.add_argument(
        "--unrate-series",
        default="UNRATE",
        help="FRED reeks voor werkloosheid (standaard: UNRATE).",
    )
    parser.add_argument(
        "--tax-series",
        default="W006RC1Q027SBEA",
        help="FRED reeks voor federal income tax revenues (standaard: W006RC1Q027SBEA).",
    )
    parser.add_argument(
        "--downturn-threshold",
        type=float,
        default=-0.05,
        help="Drempel voor downturn, bv. -0.05 voor -5%% (standaard: -0.05).",
    )
    parser.add_argument(
        "--world-bank-indicator",
        default=None,
        help="Optioneel World Bank indicator-id (jaarlijkse data, wordt naar maand geforward-filld).",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root)
    args = parse_args()

    output_path = run_pipeline(
        project_root=project_root,
        unrate_series=args.unrate_series,
        tax_series=args.tax_series,
        downturn_threshold=args.downturn_threshold,
        wb_indicator=args.world_bank_indicator,
    )

    print(f"Dataset aangemaakt: {output_path}")


if __name__ == "__main__":
    main()
