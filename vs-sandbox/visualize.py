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
TAX_SOURCES_CSV  = PROCESSED / "tax_sources.csv"

DOWNTURN_COLORS = {
    "Dot-com crash": "#ffcccc",
    "GFC":           "#ffd9b3",
    "COVID crash":   "#cce5ff",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_main() -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df


def load_lags() -> pd.DataFrame:
    return pd.read_csv(LAG_CSV)


def load_raw_events() -> pd.DataFrame:
    return pd.read_csv(RAW_EVENT_CSV)


def load_tax_sources() -> pd.DataFrame:
    df = pd.read_csv(TAX_SOURCES_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df


def shade_downturns(ax, df: pd.DataFrame) -> None:
    """Shade named downturn periods on an axis."""
    for name, color in DOWNTURN_COLORS.items():
        period = df[df["downturn_name"] == name]
        if period.empty:
            continue
        ax.axvspan(period.index.min(), period.index.max(),
                   color=color, alpha=0.4, label=name, zorder=0)


# ── plot 1: time series ───────────────────────────────────────────────────────

def plot_time_series(df: pd.DataFrame, tax: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("U.S. Market & Macroeconomic Indicators (1995–2026)",
                 fontsize=14, fontweight="bold")

    # Panel 1: S&P 500
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=1.2)
    shade_downturns(axes[0], df)
    axes[0].set_ylabel("S&P 500 (close)", fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    # Panel 2: Unemployment
    axes[1].plot(df.index, df["unemployment_rate"], color="#d62728", linewidth=1.2)
    shade_downturns(axes[1], df)
    axes[1].set_ylabel("Unemployment rate (%)", fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle="--")

    # Panel 3: Tax — FRED and Treasury as separate lines
    axes[2].plot(tax.index, tax["fred_bn"],     color="#2ca02c", linewidth=1.2,
                 label="FRED (SAAR ÷ 12, seasonally adj.)")
    axes[2].plot(tax.index, tax["treasury_bn"], color="#ff7f0e", linewidth=1.2,
                 label="Treasury MTS (seasonally adj.)")
    shade_downturns(axes[2], df)
    axes[2].set_ylabel("Federal tax receipts (bn $)", fontsize=9)
    axes[2].grid(True, alpha=0.3, linestyle="--")
    axes[2].legend(fontsize=8, loc="upper left")

    # Downturn legend on top panel
    handles = [mpatches.Patch(color=c, alpha=0.5, label=n)
               for n, c in DOWNTURN_COLORS.items()]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURES / "time_series.png", dpi=150)
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


# ── main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df   = load_main()
    lags = load_lags()
    raw  = load_raw_events()
    tax  = load_tax_sources()

    print("Generating plots...")
    plot_time_series(df, tax)
    plot_event_study(raw)
    plot_cross_correlation(lags)
    plot_heatmap(lags)

    print(f"\nAll plots saved to {FIGURES}/")


if __name__ == "__main__":
    run()