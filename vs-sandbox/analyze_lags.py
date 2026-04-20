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
LAG_OUT            = Path(__file__).resolve().parents[1] / "data" / "processed" / "lag_results.csv"
EVENT_RAW_OUT      = Path(__file__).resolve().parents[1] / "data" / "processed" / "event_study_raw.csv"
NAMED_EVENT_OUT    = Path(__file__).resolve().parents[1] / "data" / "processed" / "named_event_lags.csv"
CATALOG_OUT        = Path(__file__).resolve().parents[1] / "data" / "processed" / "downturn_catalog.csv"

MAX_LAG                 = 23
BASELINE_MONTHS         = 6      # months before downturn start used as pre-event baseline
SP500_THRESHOLD         = -19.0  # % below running all-time high to trigger a bear-market event
UNEMPLOYMENT_THRESHOLD  =  3.5   # pp absolute rise from trough to confirm unemployment increase
FED_TAX_THRESHOLD       =  7.5   # % cumulative drop from peak to confirm tax revenue decline


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


def _drawdown_series(df: pd.DataFrame) -> pd.Series:
    """Running drawdown from all-time high: (close - peak) / peak * 100."""
    close = df["close"].dropna()
    return (close - close.cummax()) / close.cummax() * 100


def detect_sp500_downturns(df: pd.DataFrame,
                           threshold: float = SP500_THRESHOLD) -> dict:
    """
    Identify one start date per bear-market cycle using peak-to-trough drawdown.

    A downturn begins the first time the drawdown crosses below `threshold`
    (default −19 %, the standard bear-market definition).  It ends when the
    index recovers to a new all-time high (drawdown returns to 0).  That entire
    decline counts as one event regardless of how volatile the path was.

    Start date = the all-time-high peak that preceded the drawdown.
    Returns a dict {label: peak_timestamp}.
    """
    close    = df["close"].dropna()
    dd       = _drawdown_series(df)
    in_down  = False
    events   = {}

    for date, val in dd.items():
        if not in_down and val <= threshold:
            in_down   = True
            peak_val  = close.cummax()[date]
            # last date where close equalled that peak (= the actual ATH date)
            peak_date = close[close == peak_val].index
            peak_date = peak_date[peak_date <= date][-1]
            label     = f"Downturn {peak_date.strftime('%b %Y')}"
            events[label] = peak_date
        elif in_down and val >= 0:
            in_down = False

    return events


NAMED_EVENTS = {
    "Dot-com crash":           ("2000-03-01", "2002-10-01"),
    "Global Financial Crisis": ("2007-10-01", "2009-03-01"),
    "COVID crash":             ("2020-02-01", "2020-04-01"),
}

# Official start dates shown in catalog / deep-dive (may differ from auto-detected peak)
NAMED_EVENT_STARTS = {
    "COVID crash": pd.Timestamp("2020-02-01"),
}


def build_downturn_catalog(df: pd.DataFrame,
                           threshold: float = SP500_THRESHOLD) -> pd.DataFrame:
    """
    For every bear-market cycle (drawdown > threshold from ATH), compute:
      - start_date        : date of the preceding all-time-high peak
      - trough_date       : lowest close within the drawdown period
      - peak_close / trough_close
      - pct_drop          : peak-to-trough %
      - duration_months   : peak to trough in months
      - name              : named label if overlaps a known event, else auto-label
      - unemp_lag_months  : first month unemployment rises ≥19 % above pre-event baseline
      - tax_lag_months    : first month tax revenue drops ≥19 % below pre-event baseline

    The trough window is the actual drawdown period (peak → next ATH recovery),
    so no arbitrary month cap is needed.
    """
    def _indicator_lag(series: pd.Series, start: pd.Timestamp, direction: str,
                       threshold: float, mom_noise_pct: float = None):
        # "up"   (unemployment, in pp): threshold = absolute pp rise from running trough
        # "down" (tax, in %):           threshold = relative % cumulative drop from running peak
        after = series[series.index > start].dropna()
        if after.empty:
            return None
        if direction == "up":
            for i in range(1, len(after)):
                window = after.iloc[:i]
                prev_min = window.min()
                if after.iloc[i] - prev_min >= threshold:   # absolute pp
                    # Last month at the trough (not first) — same as S&P uses last ATH month
                    turn_date = window[window == prev_min].index[-1]
                    lag = (turn_date.year - start.year) * 12 + (turn_date.month - start.month)
                    return lag if lag <= MAX_LAG else None
        else:
            # Find confirmation trigger: cumul >= threshold AND mom >= mom_noise_pct.
            # Trigger must fall within MAX_LAG months to exclude false positives
            # where a later-cycle peak happens to cross the threshold years later.
            trigger_i = None
            for i in range(1, len(after)):
                trigger_lag = (after.index[i].year - start.year) * 12 + (after.index[i].month - start.month)
                if trigger_lag > MAX_LAG:
                    break
                window   = after.iloc[:i]
                prev_max = window.max()
                if prev_max <= 0:
                    continue
                if (prev_max - after.iloc[i]) / prev_max * 100 >= threshold:
                    if mom_noise_pct is not None:
                        prev_val = after.iloc[i - 1]
                        mom_drop = (prev_val - after.iloc[i]) / prev_val * 100 if prev_val > 0 else 0
                        if mom_drop < mom_noise_pct:
                            continue
                    trigger_i = i
                    break

            if trigger_i is None:
                return None

            # Scan BACKWARD from the trigger to find the beginning of the decline.
            # Keep updating first_i for every month with mom >= mom_noise_pct,
            # so we end up at the EARLIEST real drop in the window.
            first_i = trigger_i
            if mom_noise_pct is not None:
                for j in range(trigger_i, 0, -1):
                    prev_val = after.iloc[j - 1]
                    if prev_val <= 0:
                        continue
                    mom_drop = (prev_val - after.iloc[j]) / prev_val * 100
                    if mom_drop >= mom_noise_pct:
                        first_i = j

            turn_date = after.index[first_i - 1]
            lag = (turn_date.year - start.year) * 12 + (turn_date.month - start.month)
            return lag if lag <= MAX_LAG else None
        return None

    close = df["close"].dropna()
    dd    = _drawdown_series(df)

    # Collect (peak_date, recovery_date) pairs for each bear market
    periods = []
    in_down    = False
    peak_date  = None

    for date, val in dd.items():
        if not in_down and val <= threshold:
            in_down   = True
            peak_val  = close.cummax()[date]
            pd_candidates = close[close == peak_val].index
            peak_date = pd_candidates[pd_candidates <= date][-1]
        elif in_down and val >= 0:
            in_down = False
            periods.append((peak_date, date))

    if in_down and peak_date is not None:          # open downturn at end of data
        periods.append((peak_date, close.index[-1]))

    rows = []
    for peak_dt, recovery_dt in periods:
        window       = close[peak_dt:recovery_dt]
        trough_date  = window.idxmin()
        peak_close   = close[peak_dt]
        trough_close = close[trough_date]
        pct_drop     = (trough_close - peak_close) / peak_close * 100
        duration     = (trough_date.year - peak_dt.year) * 12 + (trough_date.month - peak_dt.month)

        label = f"Downturn {peak_dt.strftime('%b %Y')}"
        name  = label
        for n, (ns, ne) in NAMED_EVENTS.items():
            ns_ts = pd.Timestamp(ns)
            months_from_start = (peak_dt.year - ns_ts.year) * 12 + (peak_dt.month - ns_ts.month)
            if ns_ts <= peak_dt <= pd.Timestamp(ne) or abs(months_from_start) <= 2:
                name = n
                break

        catalog_start = NAMED_EVENT_STARTS.get(name, peak_dt)
        rows.append({
            "label":            label,
            "name":             name,
            "start_date":       catalog_start,
            "trough_date":      trough_date,
            "peak_close":       round(peak_close, 2),
            "trough_close":     round(trough_close, 2),
            "pct_drop":         round(pct_drop, 2),
            "duration_months":  duration,
            "unemp_lag_months": _indicator_lag(df["unemployment_rate"], catalog_start, "up",   UNEMPLOYMENT_THRESHOLD),
            "tax_lag_months":   _indicator_lag(df["receipts_bn"],       catalog_start, "down", FED_TAX_THRESHOLD, mom_noise_pct=0.5),
        })

    # Always include named events even if their drawdown didn't reach the threshold
    covered_names = {r["name"] for r in rows}
    for name, (ns, ne) in NAMED_EVENTS.items():
        if name in covered_names:
            continue
        window = close[pd.Timestamp(ns):pd.Timestamp(ne)]
        if window.empty:
            continue
        peak_dt      = window.idxmax()
        trough_date  = window.idxmin()
        peak_close   = close[peak_dt]
        trough_close = close[trough_date]
        pct_drop     = (trough_close - peak_close) / peak_close * 100
        duration     = (trough_date.year - peak_dt.year) * 12 + (trough_date.month - peak_dt.month)
        named_start = NAMED_EVENT_STARTS.get(name, peak_dt)
        rows.append({
            "label":            f"drop_{peak_dt.strftime('%Y-%m')}",
            "name":             name,
            "start_date":       named_start,
            "trough_date":      trough_date,
            "peak_close":       round(peak_close, 2),
            "trough_close":     round(trough_close, 2),
            "pct_drop":         round(pct_drop, 2),
            "duration_months":  duration,
            "unemp_lag_months": _indicator_lag(df["unemployment_rate"], named_start, "up",   UNEMPLOYMENT_THRESHOLD),
            "tax_lag_months":   _indicator_lag(df["receipts_bn"],       named_start, "down", FED_TAX_THRESHOLD, mom_noise_pct=0.5),
        })

    catalog = pd.DataFrame(rows).sort_values("pct_drop")
    catalog.to_csv(CATALOG_OUT, index=False)
    print(f"  Saved {CATALOG_OUT.name}  ({len(catalog)} events)")
    return catalog


def all_event_starts(df: pd.DataFrame) -> dict:
    """Combined event set: drawdown-detected + named, with named labels taking priority."""
    detected = detect_sp500_downturns(df)
    named    = downturn_starts_named(df)

    def _near_named(date: pd.Timestamp) -> bool:
        """True if date falls inside a named window OR within 2 months of its start."""
        for _, (ns, ne) in NAMED_EVENTS.items():
            ns_ts, ne_ts = pd.Timestamp(ns), pd.Timestamp(ne)
            months_from_start = (date.year - ns_ts.year) * 12 + (date.month - ns_ts.month)
            if ns_ts <= date <= ne_ts or abs(months_from_start) <= 2:
                return True
        return False

    combined = {k: v for k, v in detected.items() if not _near_named(v)}
    combined.update(named)
    return combined


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
    starts = all_event_starts(df)
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


# ── named-event breakdown ─────────────────────────────────────────────────────

def run_named_event_study(df: pd.DataFrame) -> pd.DataFrame:
    """
    Event study for ALL identified downturns: auto-detected + named.
    Returns a DataFrame with columns: event, lag, unemp_change, tax_change.
    """
    rows = []
    for event, start in sorted(all_event_starts(df).items(), key=lambda x: x[1]):
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


# ── summary print ─────────────────────────────────────────────────────────────

def print_summary(agg: pd.DataFrame, xcorr: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RESEARCH QUESTION ANSWER")
    print("How long after a U.S. stock market downturn do unemployment")
    print("and federal tax revenues change?")
    print("=" * 60)

    # Primary: event study — conditional on the 9 identified downturns
    print("\n── PRIMARY: Event Study (conditional on identified downturns) ──")
    peak_u  = agg.loc[agg["avg_unemp_change"].idxmax(), "lag"]
    # First lag where avg_unemp_change turns positive
    pos_u   = agg[agg["avg_unemp_change"] > 0].iloc[0]["lag"] if not agg[agg["avg_unemp_change"] > 0].empty else None
    sig_u   = agg[agg["p_unemp"] < 0.05]
    print(f"  Unemployment first rises   : lag {int(pos_u)} months after downturn start")
    print(f"  Unemployment peaks on avg  : lag {int(peak_u)} months")
    if sig_u.empty:
        print(f"  Statistical significance   : no lag reaches p<0.05 (only {len(agg['n_events'].unique())} events)")
    else:
        print(f"  Significant lags (p<0.05)  : {sig_u['lag'].tolist()}")

    # Tax: find first lag where avg_tax_change is negative (i.e., receipts fall below baseline)
    neg_t   = agg[agg["avg_tax_change"] < 0]
    if not neg_t.empty:
        print(f"  Tax receipts first fall    : lag {int(neg_t.iloc[0]['lag'])} months")
    else:
        print(f"  Tax receipts               : avg stays above pre-event baseline across all lags")
        print(f"                               (long-term upward trend dominates; see per-event catalog)")

    print("\n  Aggregated changes by lag:")
    print(agg[["lag", "avg_unemp_change", "avg_tax_change", "p_unemp", "p_tax"]].to_string(index=False))

    # Supporting: cross-correlation — full dataset, NOT event-conditional
    print("\n── SUPPORTING: Cross-Correlation (all months, not event-conditional) ──")
    print("  Note: uses the full time series (~900 months), not just the 9 downturn events.")
    print("  Signal is diluted by normal (non-downturn) months.")
    peak_xu = xcorr.loc[xcorr["r_unemp"].abs().idxmax(), "lag"]
    peak_xt = xcorr.loc[xcorr["r_tax"].abs().idxmax(), "lag"]
    r_xu    = xcorr.loc[xcorr["lag"] == peak_xu, "r_unemp"].values[0]
    r_xt    = xcorr.loc[xcorr["lag"] == peak_xt, "r_tax"].values[0]
    print(f"  Strongest unemp correlation : lag {peak_xu} months  (r={r_xu})")
    print(f"  Strongest tax correlation   : lag {peak_xt} months  (r={r_xt})")
    print("=" * 60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading data...")
    df = load()
    print(f"  {len(df)} rows  ({df.index.min().date()} → {df.index.max().date()})")

    events = detect_sp500_downturns(df)
    print(f"  Detected {len(events)} bear-market events (drawdown threshold {SP500_THRESHOLD}%):")
    for label, start in sorted(events.items(), key=lambda x: x[1]):
        print(f"    {label}  →  start {start.date()}")

    print("Building downturn catalog...")
    build_downturn_catalog(df)

    print("Running event study...")
    raw   = run_event_study(df)
    agg   = aggregate_event_study(raw)

    print("Running named-event study (Dot-com, Global Financial Crisis, COVID)...")
    named_raw = run_named_event_study(df)
    named_raw.to_csv(NAMED_EVENT_OUT, index=False)
    print(f"  Saved {NAMED_EVENT_OUT.name}")

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