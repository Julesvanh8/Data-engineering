"""
build_dataset.py — join all ingested sources into a single clean monthly
dataset ready for lag analysis.

What this script does:
  1. Loads S&P 500, unemployment, and tax revenue from SQLite
  2. Stitches FRED (1995-2014) and Treasury (2015+) tax series
  3. Aligns everything to a common monthly index
  4. Flags named downturn events
  5. Outputs data/processed/merged_monthly_vs.csv

Usage:
    python processing/build_dataset.py
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose


DB_PATH        = Path(__file__).resolve().parents[1] / "data" / "raw" / "market_data.db"
PROCESSED_DIR  = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_PATH       = PROCESSED_DIR / "merged_monthly_vs.csv"
TAX_SOURCES_PATH  = PROCESSED_DIR / "tax_sources.csv"
PLOT_PATH         = PROCESSED_DIR / "tax_source_comparison.png"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_table(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    df = pd.read_sql_query(query, conn, parse_dates=["date"])
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df


# ── load ─────────────────────────────────────────────────────────────────────

def load_sp500(conn: sqlite3.Connection) -> pd.DataFrame:
    return load_table(conn, "SELECT date, close, pct_change FROM sp500")


def load_unemployment(conn: sqlite3.Connection) -> pd.DataFrame:
    return load_table(conn, "SELECT date, rate AS unemployment_rate FROM unemployment")


def load_tax(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Stitch FRED (pre-2015) and Treasury (2015+) into one series for analysis.
    Also saves tax_sources.csv with both full series for visualisation.
    Returns a DataFrame with columns: receipts_bn, tax_source.
    """
    df = load_table(conn, "SELECT date, receipts_bn, source FROM tax_revenue")

    fred     = df[df["source"] == "FRED_quarterly"].copy()
    treasury = df[df["source"] == "Treasury_monthly"].copy()

    # FRED W006RC1Q027SBEA is SAAR — divide by 12 for monthly equivalent.
    fred["receipts_bn"] = fred["receipts_bn"] / 12

    # Seasonally adjust Treasury so both series are comparable.
    treasury = _seasonal_adjust(treasury, "receipts_bn")

    # Save both full series for side-by-side visualisation.
    _save_tax_sources(fred, treasury)

    # Stitch: FRED pre-2015, Treasury 2015+ (Treasury wins on overlap).
    fred_stitched = fred[fred.index < "2015-01-01"]
    combined = pd.concat([fred_stitched, treasury]).sort_index()
    combined = combined.rename(columns={"source": "tax_source"})
    return combined


def _save_tax_sources(fred: pd.DataFrame, treasury: pd.DataFrame) -> None:
    """Save FRED (full range from DB) and Treasury as separate columns."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged = fred[["receipts_bn"]].rename(columns={"receipts_bn": "fred_bn"}) \
        .join(treasury[["receipts_bn"]].rename(columns={"receipts_bn": "treasury_bn"}), how="outer")
    merged.index.name = "date"
    merged.to_csv(TAX_SOURCES_PATH)
    print(f"  Saved tax_sources.csv ({len(merged)} rows)")


def _seasonal_adjust(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return df with col replaced by its seasonally-adjusted values.

    Builds a complete monthly DatetimeIndex from min→max, interpolates any
    gaps, runs seasonal_decompose, then subtracts the seasonal component
    from the original (non-interpolated) values.
    If fewer than 24 observations exist the series is returned unchanged.
    """
    original = df[col].copy()

    if original.notna().sum() < 24:
        print(f"  Skipping seasonal adjustment — fewer than 24 observations.")
        return df

    # Fill the full monthly range so seasonal_decompose gets a regular series.
    full_idx = pd.date_range(original.index.min(), original.index.max(), freq="MS")
    series_full = original.reindex(full_idx).interpolate(method="linear")

    result   = seasonal_decompose(series_full, model="additive", period=12,
                                  extrapolate_trend="freq")
    seasonal = result.seasonal

    # Subtract seasonal component from the original sparse index.
    adjusted = original - seasonal.reindex(original.index)

    df = df.copy()
    df[col] = adjusted
    print(f"  Treasury receipts seasonally adjusted (additive, period=12).")
    return df


def load_downturns(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM downturn_events", conn)


# ── stitch & align ───────────────────────────────────────────────────────────

def build_master_index(sp500: pd.DataFrame, unemployment: pd.DataFrame, tax: pd.DataFrame) -> pd.DatetimeIndex:
    """Use S&P 500 as the spine — it has the longest clean monthly series."""
    return sp500.index


def flag_downturns(df: pd.DataFrame, downturns: pd.DataFrame) -> pd.DataFrame:
    """Add a column 'downturn_name' with the event name for months inside a downturn window."""
    df["downturn_name"] = None
    for _, row in downturns.iterrows():
        mask = (df.index >= row["start_date"]) & (df.index <= row["end_date"])
        df.loc[mask, "downturn_name"] = row["name"]
    return df


# ── comparison plot ───────────────────────────────────────────────────────────

def plot_tax_comparison(tax: pd.DataFrame) -> None:
    """
    Plot FRED vs Treasury tax receipts for the overlap period (2015+)
    to validate the two sources track each other.
    """
    fred     = tax[tax["tax_source"] == "FRED_quarterly"]["receipts_bn"]
    treasury = tax[tax["tax_source"] == "Treasury_monthly"]["receipts_bn"]

    # Only plot from 2015 where both could overlap
    fred_overlap     = fred[fred.index >= "2015-01-01"]
    treasury_overlap = treasury[treasury.index >= "2015-01-01"]

    if fred_overlap.empty:
        print("  No FRED data from 2015+ to compare (expected — FRED stored pre-2015 only).")
        print("  Reloading full FRED series from API for comparison plot...")

        # For comparison only: reload full FRED without the pre-2015 cutoff
        # We do this by reading what we have and forward-filling all quarters
        all_tax = tax.copy()
        print("  Skipping comparison plot — FRED overlap data not in DB.")
        print("  Tip: run ingest_tax_fred.py with the cutoff removed to generate the comparison.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(fred_overlap.index,     fred_overlap.values,     label="FRED (W006RC1Q027SBEA, quarterly→monthly)", linestyle="--", color="steelblue")
    ax.plot(treasury_overlap.index, treasury_overlap.values, label="Treasury MTS (monthly)",                    linestyle="-",  color="darkorange")
    ax.set_title("Individual income tax receipts: FRED vs Treasury (2015–present)")
    ax.set_ylabel("USD billions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"  Comparison plot saved to {PLOT_PATH}")


# ── main ─────────────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    print("Loading sources...")
    sp500        = load_sp500(conn)
    unemployment = load_unemployment(conn)
    tax          = load_tax(conn)
    downturns    = load_downturns(conn)

    print("SP500 sample dates:", sp500.index[:3].tolist())
    print("Tax sample dates:  ", tax.index[:3].tolist())
    print("Tax 2015 dates:    ", tax[tax.index >= "2015-01-01"].index[:3].tolist())

    print(f"  S&P 500:      {len(sp500)} rows  ({sp500.index.min().date()} → {sp500.index.max().date()})")
    print(f"  Unemployment: {len(unemployment)} rows  ({unemployment.index.min().date()} → {unemployment.index.max().date()})")
    print(f"  Tax (combined): {len(tax)} rows  ({tax.index.min().date()} → {tax.index.max().date()})")

    print("\nBuilding comparison plot...")
    plot_tax_comparison(tax)

    print("\nJoining sources...")
    master_index = build_master_index(sp500, unemployment, tax)

    df = sp500.copy()
    df = df.join(unemployment,              how="left")
    df = df.join(tax[["receipts_bn",
                       "tax_source"]],      how="left")

    print("\nFlagging downturn events...")
    df = flag_downturns(df, downturns)

    # Report alignment quality
    missing_unemp = df["unemployment_rate"].isna().sum()
    missing_tax   = df["receipts_bn"].isna().sum()
    print(f"  Missing unemployment months : {missing_unemp}")
    print(f"  Missing tax months          : {missing_tax}")

    if missing_tax > 0:
        print(f"  Missing tax date range      : {df[df['receipts_bn'].isna()].index.min().date()} → {df[df['receipts_bn'].isna()].index.max().date()}")

    # Drop rows where we have no data at all for the economic series
    df = df.dropna(subset=["unemployment_rate", "receipts_bn"])

    print(f"\nFinal dataset: {len(df)} rows  ({df.index.min().date()} → {df.index.max().date()})")
    print(f"Downturn months flagged:")
    print(df["downturn_name"].value_counts(dropna=True).to_string())

    df.index.name = "date"
    df.to_csv(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    build_dataset()
