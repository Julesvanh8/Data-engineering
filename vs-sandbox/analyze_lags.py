"""
analyze_lags.py — event-study lag analysis for the vs-sandbox pipeline.

Research question:
    How long after a U.S. stock market downturn do unemployment and
    federal income tax revenues change?

Two complementary approaches:
    1. Event study   — track average change vs pre-downturn baseline
                       at each lag (1-18 months) across all named events.
    2. Cross-correlation — Pearson r between monthly S&P 500 returns
                           and economic indicator changes at each forward lag.

Outputs:
    data/processed/lag_results.csv      (aggregated per-lag statistics)
    data/processed/event_study_raw.csv  (per-event, per-lag raw changes)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH      = Path(__file__).resolve().parents[1] / "data" / "processed" / "merged_monthly_vs.csv"
LAG_OUT        = Path(__file__).resolve().parents[1] / "data" / "processed" / "lag_results.csv"
EVENT_RAW_OUT  = Path(__file__).resolve().parents[1] / "data" / "processed" / "event_study_raw.csv"

MAX_LAG           = 18
BASELINE_MONTHS   = 6    # months before downturn start used as pre-event baseline
DROP_THRESHOLD    = -5.0 # S&P monthly % change below this triggers a downturn event
CLUSTER_GAP       = 3    # consecutive drops within this many months = same event


# ── load ──────────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df


# ── event study ───────────────────────────────────────────────────────────────

def downturn_starts_named(df: pd.DataFrame) -> dict:
    """First month of each pre-seeded named downturn event."""
    events = {}
    for name, group in df[df["downturn_name"].notna()].groupby("downturn_name"):
        events[name] = group.index.min()
    return events


def detect_sp500_downturns(df: pd.DataFrame,
                           threshold: float = DROP_THRESHOLD,
                           gap: int = CLUSTER_GAP) -> dict:
    """
    Identify downturn starts from S&P 500 monthly returns.

    A 'downturn start' is the first month of a cluster of months where
    pct_change < threshold.  Months within `gap` months of the previous
    drop are merged into the same event so one crash = one event.

    Returns a dict {label: start_timestamp}.
    """
    drops = df[df["pct_change"] < threshold].index.sort_values()
    if drops.empty:
        return {}

    events = {}
    cluster_start = drops[0]
    prev = drops[0]

    for date in drops[1:]:
        months_apart = (date.year - prev.year) * 12 + (date.month - prev.month)
        if months_apart > gap:
            label = f"drop_{cluster_start.strftime('%Y-%m')}"
            events[label] = cluster_start
            cluster_start = date
        prev = date

    events[f"drop_{cluster_start.strftime('%Y-%m')}"] = cluster_start
    return events


def pre_event_baseline(df: pd.DataFrame, start: pd.Timestamp) -> dict:
    """Mean of unemployment and tax over the BASELINE_MONTHS before start."""
    window = df[df.index < start].tail(BASELINE_MONTHS)
    return {
        "unemployment_rate": window["unemployment_rate"].mean(),
        "receipts_bn":       window["receipts_bn"].mean(),
    }


def run_event_study(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each detected S&P 500 downturn and each lag 1-MAX_LAG, compute the
    absolute change from the pre-event baseline in unemployment rate and tax receipts.
    """
    starts = detect_sp500_downturns(df)
    rows = []
    for event, start in starts.items():
        base = pre_event_baseline(df, start)
        for lag in range(1, MAX_LAG + 1):
            target = (start + pd.DateOffset(months=lag)).to_period("M").to_timestamp()
            if target not in df.index:
                continue
            obs = df.loc[target]
            rows.append({
                "event":        event,
                "lag":          lag,
                "unemp_change": obs["unemployment_rate"] - base["unemployment_rate"],
                "tax_change":   obs["receipts_bn"]       - base["receipts_bn"],
            })
    return pd.DataFrame(rows)


def aggregate_event_study(raw: pd.DataFrame) -> pd.DataFrame:
    """Average changes across events per lag, with a one-sample t-test vs 0."""
    rows = []
    for lag in range(1, MAX_LAG + 1):
        sub = raw[raw["lag"] == lag]
        if sub.empty:
            continue

        u = sub["unemp_change"].dropna()
        t = sub["tax_change"].dropna()

        _, p_u = stats.ttest_1samp(u, 0) if len(u) > 1 else (np.nan, np.nan)
        _, p_t = stats.ttest_1samp(t, 0) if len(t) > 1 else (np.nan, np.nan)

        rows.append({
            "lag":              lag,
            "avg_unemp_change": u.mean(),
            "std_unemp_change": u.std(),
            "avg_tax_change":   t.mean(),
            "std_tax_change":   t.std(),
            "n_events":         len(u),
            "p_unemp":          round(p_u, 4),
            "p_tax":            round(p_t, 4),
        })
    return pd.DataFrame(rows)


# ── cross-correlation ─────────────────────────────────────────────────────────

def run_cross_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation between S&P 500 monthly return and the change in
    unemployment / tax receipts at each forward lag (1-MAX_LAG months).

    A negative sp_return correlated with a positive unemp_change at lag k
    means: bad months for the market predict rising unemployment k months later.
    """
    work = df.copy()
    work["unemp_change"] = work["unemployment_rate"].diff()
    work["tax_change"]   = work["receipts_bn"].diff()
    sp = work["pct_change"].dropna()

    rows = []
    for lag in range(1, MAX_LAG + 1):
        future_u = work["unemp_change"].shift(-lag)
        future_t = work["tax_change"].shift(-lag)

        aligned_u = pd.concat([sp, future_u], axis=1, sort=False).dropna()
        aligned_t = pd.concat([sp, future_t], axis=1, sort=False).dropna()

        r_u, p_u = stats.pearsonr(aligned_u.iloc[:, 0], aligned_u.iloc[:, 1]) if len(aligned_u) > 2 else (np.nan, np.nan)
        r_t, p_t = stats.pearsonr(aligned_t.iloc[:, 0], aligned_t.iloc[:, 1]) if len(aligned_t) > 2 else (np.nan, np.nan)

        rows.append({
            "lag":    lag,
            "r_unemp": round(r_u, 4),
            "p_unemp": round(p_u, 4),
            "r_tax":   round(r_t, 4),
            "p_tax":   round(p_t, 4),
        })
    return pd.DataFrame(rows)


# ── summary print ─────────────────────────────────────────────────────────────

def print_summary(agg: pd.DataFrame, xcorr: pd.DataFrame) -> None:
    peak_u = agg.loc[agg["avg_unemp_change"].idxmax(), "lag"]
    peak_t = agg.loc[agg["avg_tax_change"].abs().idxmax(), "lag"]

    print("\n" + "=" * 60)
    print("LAG ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nEvent study — peak unemployment change : lag {peak_u} months")
    print(f"Event study — peak tax receipt change  : lag {peak_t} months")

    print("\nEvent study aggregated (avg change from pre-downturn baseline):")
    print(agg[["lag", "avg_unemp_change", "avg_tax_change", "p_unemp", "p_tax"]].to_string(index=False))

    peak_xu = xcorr.loc[xcorr["r_unemp"].abs().idxmax(), "lag"]
    peak_xt = xcorr.loc[xcorr["r_tax"].abs().idxmax(), "lag"]
    print(f"\nCross-correlation — strongest unemp correlation : lag {peak_xu} months  (r={xcorr.loc[xcorr['lag']==peak_xu,'r_unemp'].values[0]})")
    print(f"Cross-correlation — strongest tax correlation   : lag {peak_xt} months  (r={xcorr.loc[xcorr['lag']==peak_xt,'r_tax'].values[0]})")
    print("=" * 60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading data...")
    df = load()
    print(f"  {len(df)} rows  ({df.index.min().date()} → {df.index.max().date()})")

    events = detect_sp500_downturns(df)
    print(f"  Detected {len(events)} downturn events from S&P 500 (threshold {DROP_THRESHOLD}%):")
    for label, start in sorted(events.items(), key=lambda x: x[1]):
        print(f"    {label}  →  start {start.date()}")

    print("Running event study...")
    raw   = run_event_study(df)
    agg   = aggregate_event_study(raw)

    print("Running cross-correlation analysis...")
    xcorr = run_cross_correlation(df)

    # Merge into one output file
    results = agg.merge(xcorr, on="lag", suffixes=("_event", "_xcorr"))
    results.to_csv(LAG_OUT, index=False)
    raw.to_csv(EVENT_RAW_OUT, index=False)
    print(f"  Saved {LAG_OUT.name}")
    print(f"  Saved {EVENT_RAW_OUT.name}")

    print_summary(agg, xcorr)
    return df, results, raw


if __name__ == "__main__":
    run()