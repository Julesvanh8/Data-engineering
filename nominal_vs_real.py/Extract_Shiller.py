from __future__ import annotations

from pathlib import Path
import io

import pandas as pd
import requests


SP500_DATA_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"


def ensure_raw_directory(project_root: Path) -> Path:
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def fetch_sp500_shiller_robust() -> pd.DataFrame:
    """
    Download the GitHub S&P 500 dataset and keep the original pipeline columns,
    plus the variables needed for the robustness check.

    Original pipeline columns:
    - close
    - adjusted_close
    - volume

    Extra robustness columns:
    - real_close
    - cpi
    """

    response = requests.get(SP500_DATA_URL, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df.columns = [str(col).strip() for col in df.columns]

    required_original_columns = ["Date", "SP500"]

    missing_original_columns = [
        col for col in required_original_columns
        if col not in df.columns
    ]

    if missing_original_columns:
        raise ValueError(
            f"Deze verplichte kolommen ontbreken: {missing_original_columns}. "
            f"Beschikbare kolommen zijn: {list(df.columns)}"
        )

    rename_map = {
        "Date": "date",
        "SP500": "close",
        "Consumer Price Index": "cpi",
        "Real Price": "real_close",
    }

    available_rename_map = {
        old_name: new_name
        for old_name, new_name in rename_map.items()
        if old_name in df.columns
    }

    df = df.rename(columns=available_rename_map)

    required_robust_columns = ["real_close", "cpi"]

    missing_robust_columns = [
        col for col in required_robust_columns
        if col not in df.columns
    ]

    if missing_robust_columns:
        raise ValueError(
            f"Deze robustness kolommen ontbreken: {missing_robust_columns}. "
            f"Beschikbare kolommen zijn: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "close",
        "real_close",
        "cpi",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close", "real_close", "cpi"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset="date", keep="first")

    # Keep compatibility with the original pipeline
    df["adjusted_close"] = df["close"]
    df["volume"] = pd.NA

    columns_to_keep = [
        "date",
        "close",
        "adjusted_close",
        "volume",
        "real_close",
        "cpi",
    ]

    return df[columns_to_keep]


def main() -> None:
    try:
        project_root = Path(__file__).resolve().parents[1]
    except NameError:
        project_root = Path.cwd()

    raw_dir = ensure_raw_directory(project_root)
    output_file = raw_dir / "github_sp500_shiller_robust.csv"

    print("Downloading GitHub S&P 500 Shiller-style data...")

    df = fetch_sp500_shiller_robust()
    df.to_csv(output_file, index=False)

    print(f"Bestand opgeslagen als: {output_file}")
    print(f"Aantal observaties: {len(df)}")
    print(f"Periode: {df['date'].min().date()} tot {df['date'].max().date()}")
    print("Kolommen:", df.columns.tolist())
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()