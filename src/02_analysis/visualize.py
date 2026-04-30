"""
visualize.py — static plots for the lag analysis.

Plots produced (saved to outputs/figures/):
  1. time_series.png       — S&P 500, unemployment, and tax receipts over time
                             with named downturn periods shaded.
  2. event_study.png       — Impulse-response style: per-event lines + average
                             change in unemployment and tax at each lag.
  3. named_event_lags.png  — Per-named-event lag analysis.

Usage:
    python vs-sandbox/visualize.py
    (run analyze_lags.py first to generate the CSV inputs)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sqlite3

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
FIGURES   = Path(__file__).resolve().parents[2] / "outputs" / "figures"

DB_PATH    = Path(__file__).resolve().parents[2] / "data" / "raw" / "main_marts.db"
EVENTS_CSV = PROCESSED / "events_combined.csv"

FRED_UNRATE_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "fred_unrate.csv"

NAMED_EVENT_NAMES = {"Dot-com crash", "Global Financial Crisis", "COVID crash"}

NAMED_COLORS = {
    "Dot-com crash":           "#ffcccc",
    "Global Financial Crisis": "#ffd9b3",
    "COVID crash":             "#cce5ff",
}
OTHER_COLOR = "#e0e0e0"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_main() -> pd.DataFrame:
    """Load data from DBT mart."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM fct_combined_monthly ORDER BY date",
        conn,
        parse_dates=["date"],
        index_col="date"
    )
    conn.close()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df


def load_raw_events() -> pd.DataFrame:
    return pd.read_csv(EVENTS_CSV)


def load_named_events() -> pd.DataFrame:
    df = pd.read_csv(EVENTS_CSV)
    return df[df["name"].isin(NAMED_EVENT_NAMES)]


def load_fred_unrate() -> pd.Series:
    df = pd.read_csv(FRED_UNRATE_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df["UNRATE"]


def load_catalog() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        return pd.DataFrame(columns=["name", "start_date", "trough_date", "pct_drop", "duration_months"])
    df = pd.read_csv(EVENTS_CSV, parse_dates=["sp500_start", "sp500_trough"])
    return (
        df.drop_duplicates(subset=["name"])
        .rename(columns={
            "sp500_start":    "start_date",
            "sp500_trough":   "trough_date",
            "sp500_pct_drop": "pct_drop",
            "sp500_duration": "duration_months",
        })[["name", "start_date", "trough_date", "pct_drop", "duration_months"]]
        .reset_index(drop=True)
    )


def shade_downturns(ax, catalog: pd.DataFrame) -> None:
    for _, row in catalog.iterrows():
        color = NAMED_COLORS.get(row["name"], OTHER_COLOR)
        ax.axvspan(row["start_date"], row["trough_date"],
                   color=color, alpha=0.4, zorder=0)


# ── plot 1: time series ───────────────────────────────────────────────────────

def plot_time_series(df: pd.DataFrame, fred_unrate: pd.Series,
                     catalog: pd.DataFrame) -> None:
    import numpy as np
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle("U.S. Market & Macroeconomic Indicators (1871–2026)",
                 fontsize=14, fontweight="bold")

    # Panel 1: S&P 500
    line_sp, = axes[0].plot(df.index, df["sp500_close"], color="#1f77b4", linewidth=1.2)
    shade_downturns(axes[0], catalog)
    axes[0].set_ylabel("S&P 500 (close)", fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    named_patches = [mpatches.Patch(color=c, alpha=0.5, label=n)
                     for n, c in NAMED_COLORS.items()]
    other_patch = mpatches.Patch(color=OTHER_COLOR, alpha=0.5, label="Other detected downturns")
    axes[0].legend(handles=[line_sp] + named_patches + [other_patch],
                   fontsize=8, loc="upper left")

    # Panel 2: S&P 500 log scale
    line_log, = axes[1].plot(df.index, np.log(df["sp500_close"]), color="#1f77b4", linewidth=1.2)
    shade_downturns(axes[1], catalog)
    axes[1].set_ylabel("log(S&P 500)", fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(handles=[line_log], labels=["log(S&P 500)"], fontsize=8, loc="upper left")

    # Panel 3: Unemployment
    line_unemp, = axes[2].plot(fred_unrate.index, fred_unrate,
                               color="#d62728", linewidth=1.2)
    shade_downturns(axes[2], catalog)
    axes[2].set_ylabel("Unemployment rate (%)", fontsize=9)
    axes[2].grid(True, alpha=0.3, linestyle="--")
    axes[2].legend(handles=[line_unemp], labels=["FRED UNRATE"],
                   fontsize=8, loc="upper left")

    # Panel 4: Tax receipts
    line_tax, = axes[3].plot(df.index, df["federal_tax_revenue"], color="#2ca02c", linewidth=1.2)
    shade_downturns(axes[3], catalog)
    axes[3].set_ylabel("Federal tax receipts (bn $)", fontsize=9)
    axes[3].grid(True, alpha=0.3, linestyle="--")
    axes[3].legend(handles=[line_tax],
                   labels=["FRED W006RC1Q027SBEA (SAAR ÷ 12, seasonally adj.)"],
                   fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURES / "time_series.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved time_series.png")


# ── plot 2: event study ───────────────────────────────────────────────────────

def plot_event_study(raw: pd.DataFrame) -> None:
    fig, (ax_u, ax_t) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Event Study: Average Change After S&P 500 Downturn Start",
                 fontsize=13, fontweight="bold")

    events = raw["name"].unique()
    lags   = sorted(raw["lag"].unique())

    for event in events:
        sub = raw[raw["name"] == event].set_index("lag").reindex(lags)
        ax_u.plot(lags, sub["unemp_change"], color="gray",  alpha=0.3, linewidth=0.8)
        ax_t.plot(lags, sub["tax_change"],   color="gray",  alpha=0.3, linewidth=0.8)

    avg_u = raw.groupby("lag")["unemp_change"].mean()
    avg_t = raw.groupby("lag")["tax_change"].mean()

    ax_u.plot(lags, avg_u.reindex(lags), color="#d62728", linewidth=2.5,
              marker="o", markersize=5, label="Average")
    ax_t.plot(lags, avg_t.reindex(lags), color="#2ca02c", linewidth=2.5,
              marker="o", markersize=5, label="Average")

    for ax in (ax_u, ax_t):
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Months after downturn start")
        ax.set_xticks(lags)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()

    ax_u.set_title("Unemployment rate change (pp vs baseline)")
    ax_u.set_ylabel("Percentage-point change")
    ax_t.set_title("Tax receipts change (% vs baseline)")
    ax_t.set_ylabel("% change")

    fig.tight_layout()
    fig.savefig(FIGURES / "event_study.png", dpi=150)
    plt.close()
    print("  Saved event_study.png")


# ── plot 3: per-named-event lag analysis ──────────────────────────────────────

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f"]

def plot_named_event_lags(named: pd.DataFrame) -> None:
    events = sorted(named["name"].unique(),
                    key=lambda e: named[named["name"] == e]["lag"].count(), reverse=True)
    colors = {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(events)}

    fig, (ax_u, ax_t) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Lag Analysis per Downturn Event\n"
                 "(change vs 6-month pre-event baseline)",
                 fontsize=13, fontweight="bold")

    lags = sorted(named["lag"].unique())

    for event in events:
        color = colors[event]
        sub = named[named["name"] == event].set_index("lag").reindex(lags)
        ax_u.plot(lags, sub["unemp_change"], color=color, linewidth=2,
                  marker="o", markersize=5, label=event)
        ax_t.plot(lags, sub["tax_change"],   color=color, linewidth=2,
                  marker="o", markersize=5, label=event)

    for ax in (ax_u, ax_t):
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Months after downturn start")
        ax.set_xticks(lags)
        ax.grid(True, alpha=0.3, linestyle="--")

    ax_u.legend(fontsize=9, loc="upper right")
    ax_t.legend(fontsize=9, loc="lower left", ncol=2)

    ax_u.set_title("Unemployment rate change (pp vs baseline)")
    ax_u.set_ylabel("Percentage-point change")
    ax_t.set_title("Tax receipts change (% vs baseline)")
    ax_t.set_ylabel("% change")

    fig.tight_layout()
    fig.savefig(FIGURES / "named_event_lags.png", dpi=150)
    plt.close()
    print("  Saved named_event_lags.png")


# ── main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df         = load_main()
    raw        = load_raw_events()
    named      = load_named_events()
    fred_unemp = load_fred_unrate()
    catalog    = load_catalog()

    if not catalog.empty:
        print("\nDetected downturns (sorted by % drop):")
        print(catalog[["name", "start_date", "trough_date", "pct_drop", "duration_months"]]
              .to_string(index=False))

    print("\nGenerating plots...")
    plot_time_series(df, fred_unemp, catalog)
    plot_event_study(raw)
    plot_named_event_lags(named)

    print(f"\nAll plots saved to {FIGURES}/")


if __name__ == "__main__":
    run()
