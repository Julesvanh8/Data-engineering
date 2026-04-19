"""
visualize.py — static plots for the lag analysis.

Plots produced (saved to outputs/figures/):
  1. time_series.png       — S&P 500, unemployment, and tax receipts over time
                             with named downturn periods shaded.
  2. event_study.png       — Impulse-response style: per-event lines + average
                             change in unemployment and tax at each lag.
  3. cross_correlation.png — Pearson r between S&P returns and economic
                             indicators at each forward lag (bars, p<0.05 starred).
  4. heatmap.png           — Correlation coefficients as a 2×18 heatmap.

Usage:
    python vs-sandbox/visualize.py
    (run analyze_lags.py first to generate the CSV inputs)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES   = Path(__file__).resolve().parents[1] / "outputs" / "figures"

MAIN_CSV         = PROCESSED / "merged_monthly_vs.csv"
LAG_CSV          = PROCESSED / "lag_results.csv"
RAW_EVENT_CSV    = PROCESSED / "event_study_raw.csv"
NAMED_EVENT_CSV  = PROCESSED / "named_event_lags.csv"
TAX_SOURCES_CSV  = PROCESSED / "tax_sources.csv"
FRED_UNRATE_CSV  = Path(__file__).resolve().parents[1] / "data" / "raw" / "fred_unrate.csv"
FRIEND_SP500_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "github_sp500_daily.csv"
CATALOG_CSV      = PROCESSED / "downturn_catalog.csv"

NAMED_COLORS = {
    "Dot-com crash":           "#ffcccc",
    "Global Financial Crisis": "#ffd9b3",
    "COVID crash":             "#cce5ff",
}
OTHER_COLOR = "#e0e0e0"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_main() -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df


def load_lags() -> pd.DataFrame:
    return pd.read_csv(LAG_CSV)


def load_raw_events() -> pd.DataFrame:
    return pd.read_csv(RAW_EVENT_CSV)


def load_named_events() -> pd.DataFrame:
    return pd.read_csv(NAMED_EVENT_CSV)


def load_friend_sp500() -> pd.Series:
    df = pd.read_csv(FRIEND_SP500_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df["close"]


def load_fred_unrate() -> pd.Series:
    df = pd.read_csv(FRED_UNRATE_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df["UNRATE"]


def load_tax_sources() -> pd.DataFrame:
    df = pd.read_csv(TAX_SOURCES_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df


def load_catalog() -> pd.DataFrame:
    if not CATALOG_CSV.exists():
        return pd.DataFrame(columns=["name", "start_date", "trough_date", "pct_drop", "duration_months"])
    df = pd.read_csv(CATALOG_CSV, parse_dates=["start_date", "trough_date"])
    return df


def shade_downturns(ax, catalog: pd.DataFrame) -> None:
    """Shade all catalog downturns: named ones in their color, others in gray."""
    for _, row in catalog.iterrows():
        color = NAMED_COLORS.get(row["name"], OTHER_COLOR)
        ax.axvspan(row["start_date"], row["trough_date"],
                   color=color, alpha=0.4, zorder=0)


# ── plot 1: time series ───────────────────────────────────────────────────────

def plot_time_series(df: pd.DataFrame, tax: pd.DataFrame, fred_unrate: pd.Series,
                     catalog: pd.DataFrame) -> None:
    import numpy as np
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle("U.S. Market & Macroeconomic Indicators (1871–2026)",
                 fontsize=14, fontweight="bold")

    # Panel 1: S&P 500
    line_sp, = axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=1.2)
    shade_downturns(axes[0], catalog)
    axes[0].set_ylabel("S&P 500 (close)", fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    named_patches = [mpatches.Patch(color=c, alpha=0.5, label=n)
                     for n, c in NAMED_COLORS.items()]
    other_patch   = mpatches.Patch(color=OTHER_COLOR, alpha=0.5, label="Other detected downturns")
    axes[0].legend(handles=[line_sp] + named_patches + [other_patch],
                   fontsize=8, loc="upper left")

    # Panel 2: S&P 500 log scale
    line_log, = axes[1].plot(df.index, np.log(df["close"]), color="#1f77b4", linewidth=1.2)
    shade_downturns(axes[1], catalog)
    axes[1].set_ylabel("log(S&P 500)", fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(handles=[line_log], labels=["log(S&P 500)"], fontsize=8, loc="upper left")

    # Panel 3: Unemployment — FRED UNRATE only
    line_unemp, = axes[2].plot(fred_unrate.index, fred_unrate,
                               color="#d62728", linewidth=1.2)
    shade_downturns(axes[2], catalog)
    axes[2].set_ylabel("Unemployment rate (%)", fontsize=9)
    axes[2].grid(True, alpha=0.3, linestyle="--")
    axes[2].legend(handles=[line_unemp], labels=["FRED UNRATE"],
                   fontsize=8, loc="upper left")

    # Panel 4: Tax — FRED only
    line_fred, = axes[3].plot(tax.index, tax["fred_bn"], color="#2ca02c", linewidth=1.2)
    shade_downturns(axes[3], catalog)
    axes[3].set_ylabel("Federal tax receipts (bn $)", fontsize=9)
    axes[3].grid(True, alpha=0.3, linestyle="--")
    axes[3].legend(handles=[line_fred],
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

    events = raw["event"].unique()
    lags   = sorted(raw["lag"].unique())

    for event in events:
        sub = raw[raw["event"] == event].set_index("lag").reindex(lags)
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
    ax_t.set_title("Tax receipts change (bn $ vs baseline)")
    ax_t.set_ylabel("Change in bn $")

    fig.tight_layout()
    fig.savefig(FIGURES / "event_study.png", dpi=150)
    plt.close()
    print("  Saved event_study.png")


# ── plot 3: cross-correlation ─────────────────────────────────────────────────

def plot_cross_correlation(lags: pd.DataFrame) -> None:
    fig, (ax_u, ax_t) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Cross-Correlation: S&P 500 Monthly Return → Economic Indicators",
                 fontsize=13, fontweight="bold")

    lag_vals = lags["lag"].values

    def bar_colors(r_col, p_col):
        return ["#d62728" if (r > 0 and p < 0.05) else
                "#1f77b4" if (r < 0 and p < 0.05) else
                "#aec7e8"
                for r, p in zip(lags[r_col], lags[p_col])]

    ax_u.bar(lag_vals, lags["r_unemp"], color=bar_colors("r_unemp", "p_unemp_xcorr"), edgecolor="white")
    ax_t.bar(lag_vals, lags["r_tax"],   color=bar_colors("r_tax",   "p_tax_xcorr"),   edgecolor="white")

    # Stars for significant bars
    for ax, r_col, p_col in [(ax_u, "r_unemp", "p_unemp_xcorr"),
                              (ax_t, "r_tax",   "p_tax_xcorr")]:
        for _, row in lags.iterrows():
            if row[p_col] < 0.05:
                y = row[r_col] + (0.005 if row[r_col] >= 0 else -0.012)
                ax.text(row["lag"], y, "*", ha="center", fontsize=12, color="black")

    for ax in (ax_u, ax_t):
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(lag_vals)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.set_ylabel("Pearson r")

    ax_u.set_title("Unemployment rate change (lagged)")
    ax_t.set_title("Tax receipts change (lagged)")
    ax_t.set_xlabel("Lag (months)")

    sig_patch   = mpatches.Patch(color="#1f77b4", label="Significant (p < 0.05)")
    insig_patch = mpatches.Patch(color="#aec7e8", label="Not significant")
    ax_u.legend(handles=[sig_patch, insig_patch], fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "cross_correlation.png", dpi=150)
    plt.close()
    print("  Saved cross_correlation.png")


# ── plot 4: heatmap ───────────────────────────────────────────────────────────

def plot_heatmap(lags: pd.DataFrame) -> None:
    heat = lags.set_index("lag")[["r_unemp", "r_tax"]].T
    heat.index = ["Unemployment", "Tax receipts"]

    fig, ax = plt.subplots(figsize=(14, 3))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Pearson r", "shrink": 0.6},
                ax=ax)
    ax.set_title("Cross-Correlation Heatmap: S&P 500 Return → Lagged Economic Change",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "heatmap.png", dpi=150)
    plt.close()
    print("  Saved heatmap.png")


# ── plot 5: per-named-event lag analysis ──────────────────────────────────────

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f"]

def plot_named_event_lags(named: pd.DataFrame) -> None:
    events = sorted(named["event"].unique(),
                    key=lambda e: named[named["event"] == e]["lag"].count(), reverse=True)
    colors = {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(events)}

    fig, (ax_u, ax_t) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Lag Analysis per Downturn Event\n"
                 "(change vs 6-month pre-event baseline)",
                 fontsize=13, fontweight="bold")

    lags = sorted(named["lag"].unique())

    for event in events:
        color = colors[event]
        sub = named[named["event"] == event].set_index("lag").reindex(lags)
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
    ax_t.set_title("Tax receipts change (bn $ vs baseline)")
    ax_t.set_ylabel("Change in bn $")

    fig.tight_layout()
    fig.savefig(FIGURES / "named_event_lags.png", dpi=150)
    plt.close()
    print("  Saved named_event_lags.png")

# ── main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df           = load_main()
    lags         = load_lags()
    raw          = load_raw_events()
    named        = load_named_events()
    tax          = load_tax_sources()
    fred_unemp   = load_fred_unrate()
    friend_sp500 = load_friend_sp500()
    catalog      = load_catalog()

    if not catalog.empty:
        print("\nDetected downturns (sorted by % drop):")
        print(catalog[["name", "start_date", "trough_date", "pct_drop", "duration_months"]]
              .to_string(index=False))

    print("\nGenerating plots...")
    plot_time_series(df, tax, fred_unemp, catalog)
    plot_event_study(raw)
    plot_named_event_lags(named)
    plot_cross_correlation(lags)
    plot_heatmap(lags)

    print(f"\nAll plots saved to {FIGURES}/")


if __name__ == "__main__":
    run()