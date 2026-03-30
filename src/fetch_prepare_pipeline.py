from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2/country/USA/indicator"


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


def is_present_api_key(api_key: str) -> bool:
    return bool(api_key and api_key.strip() and api_key.strip() != "Type hier je API key")


def fetch_alpha_vantage_daily(symbol: str, api_key: str) -> pd.DataFrame:
    def _request_series(function_name: str, outputsize: str) -> dict:
        params = {
            "function": function_name,
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": api_key,
        }

        last_payload: dict = {}
        for attempt in range(3):
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            last_payload = payload

            info = str(payload.get("Information", "")).lower()
            if "please consider spreading out your free api requests" in info:
                time.sleep(1.2 * (attempt + 1))
                continue

            return payload

        return last_payload

    payload = _request_series("TIME_SERIES_DAILY_ADJUSTED", outputsize="full")
    if "Time Series (Daily)" not in payload and "premium endpoint" in str(payload.get("Information", "")).lower():
        time.sleep(1.2)
        payload = _request_series("TIME_SERIES_DAILY", outputsize="compact")

    if "Time Series (Daily)" not in payload:
        raise ValueError(f"Onverwachte Alpha Vantage response: {payload}")

    ts = payload["Time Series (Daily)"]
    frame = (
        pd.DataFrame.from_dict(ts, orient="index")
        .rename(
            columns={
                "5. adjusted close": "adjusted_close",
                "4. close": "close",
                "6. volume": "volume",
                "5. volume": "volume",
            }
        )
    )

    if "adjusted_close" not in frame.columns:
        frame["adjusted_close"] = frame["close"]
    frame = frame.loc[:, ["close", "adjusted_close", "volume"]]

    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame.astype(float)
    frame.index.name = "date"
    return frame


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


def fetch_stock_daily_with_fallback(
    symbol: str,
    alpha_key: str,
    fred_key: str,
    stock_source: str,
    fred_stock_series: str,
    min_alpha_observations: int = 500,
) -> tuple[pd.DataFrame, str]:
    source = stock_source.lower()

    if source not in {"auto", "alpha", "fred"}:
        raise ValueError("--stock-source moet een van deze waarden zijn: auto, alpha, fred")

    if source in {"auto", "alpha"} and is_present_api_key(alpha_key):
        try:
            stock_daily = fetch_alpha_vantage_daily(symbol=symbol, api_key=alpha_key)
            if source == "auto" and len(stock_daily) < min_alpha_observations:
                print(
                    f"Alpha Vantage bevat slechts {len(stock_daily)} dagelijkse observaties; "
                    "fallback naar FRED voor langere historiek."
                )
            else:
                return stock_daily, "alpha_vantage"
        except ValueError as exc:
            if source == "alpha":
                raise
            print(f"Alpha Vantage niet bruikbaar, fallback naar FRED. Reden: {exc}")

    stock_daily = fetch_fred_stock_proxy(series_id=fred_stock_series, api_key=fred_key)
    return stock_daily, f"fred_{fred_stock_series.lower()}"


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


def run_pipeline(
    project_root: Path,
    symbol: str,
    unrate_series: str,
    tax_series: str,
    downturn_threshold: float,
    stock_source: str,
    fred_stock_series: str,
    wb_indicator: Optional[str],
) -> Path:
    raw_dir, processed_dir = ensure_directories(project_root)

    alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY", ALPHA_VANTAGE_API_KEY)
    fred_key = os.getenv("FRED_API_KEY", FRED_API_KEY)

    validate_api_key(fred_key, "FRED_API_KEY")

    stock_daily, stock_data_source = fetch_stock_daily_with_fallback(
        symbol=symbol,
        alpha_key=alpha_key,
        fred_key=fred_key,
        stock_source=stock_source,
        fred_stock_series=fred_stock_series,
    )
    stock_daily.to_csv(raw_dir / f"{stock_data_source}_{symbol.lower()}_daily.csv")

    unrate = fetch_fred_series(unrate_series, api_key=fred_key)
    unrate.to_frame().to_csv(raw_dir / f"fred_{unrate_series.lower()}.csv")

    tax_revenue = fetch_fred_series(tax_series, api_key=fred_key)
    tax_revenue.to_frame().to_csv(raw_dir / f"fred_{tax_series.lower()}.csv")

    stock_monthly = create_monthly_stock_features(stock_daily, downturn_threshold=downturn_threshold)
    macro_monthly = harmonize_to_monthly(unrate=unrate, tax_revenue=tax_revenue)
    macro_monthly = macro_monthly.reindex(macro_monthly.index.union(stock_monthly.index)).sort_index().ffill()
    macro_monthly = macro_monthly.reindex(stock_monthly.index)

    merged = pd.concat([stock_monthly, macro_monthly], axis=1, sort=False).dropna(how="any")
    merged = add_lag_features(merged, max_lag=12)
    merged["stock_data_source"] = stock_data_source

    if wb_indicator:
        wb_series = fetch_world_bank_indicator(wb_indicator)
        wb_monthly = wb_series.resample("ME").ffill().rename(f"world_bank_{wb_indicator.lower()}")
        wb_monthly.to_frame().to_csv(raw_dir / f"world_bank_{wb_indicator.lower()}.csv")
        merged = merged.join(wb_monthly, how="left")

    merged.index.name = "month"
    output_path = processed_dir / "merged_monthly_dataset.csv"
    merged.to_csv(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Haal data op uit Alpha Vantage en FRED, definieer downturns en maak een geharmoniseerde maanddataset."
        )
    )
    parser.add_argument("--symbol", default="SPY", help="Aandelenmarkt proxy ticker (standaard: SPY).")
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
        default=-0.10,
        help="Drempel voor downturn, bv. -0.10 voor -10%% (standaard: -0.10).",
    )
    parser.add_argument(
        "--stock-source",
        default="auto",
        choices=["auto", "alpha", "fred"],
        help="Bron voor aandelenreeks: auto (eerst Alpha, anders FRED), alpha, of fred.",
    )
    parser.add_argument(
        "--fred-stock-series",
        default="SP500",
        help="FRED reeks voor aandelenproxy bij fallback of expliciete fred-bron (standaard: SP500).",
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
        symbol=args.symbol,
        unrate_series=args.unrate_series,
        tax_series=args.tax_series,
        downturn_threshold=args.downturn_threshold,
        stock_source=args.stock_source,
        fred_stock_series=args.fred_stock_series,
        wb_indicator=args.world_bank_indicator,
    )

    print(f"Dataset aangemaakt: {output_path}")


if __name__ == "__main__":
    main()
