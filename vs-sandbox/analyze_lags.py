"""
analyze_lags.py — event-study lag analysis for the vs-sandbox pipeline.

Research question:
    How long after a U.S. stock market downturn do unemployment and
    federal income tax revenues change?

Two complementary approaches:
    1. Event study   — track average change vs pre-downturn baseline
                       at each lag (1-MAX_LAG months) across all detected events.
    2. Cross-correlation — Pearson r between monthly S&P 500 returns
                           and economic indicator changes at each forward lag.

Outputs:
    data/processed/lag_results.csv      (aggregated per-lag statistics)
    data/processed/events_combined.csv  (per-event, per-lag raw changes + catalog metadata)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH  = Path(__file__).resolve().parents[1] / "data" / "processed" / "merged_monthly_vs.csv"
LAG_OUT    = Path(__file__).resolve().parents[1] / "data" / "processed" / "lag_results.csv"
EVENTS_OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "events_combined.csv"

MAX_LAG                 = 23
SP500_THRESHOLD         = -19.0  # % fall from running all-time high to trigger a bear-market event
UNEMPLOYMENT_THRESHOLD  =  2     # pp absolute rise from trough to confirm unemployment increase
FED_TAX_THRESHOLD       = -7.5   # % cumulative drop from peak to confirm tax revenue decline
MOM_NOISE_PCT           =  1.5   # % min month-to-month drop to count as a real tax decline move

NAMED_EVENTS = {
    "Dot-com crash":           ("2000-03-01", "2002-10-01"),
    "Global Financial Crisis": ("2007-10-01", "2009-03-01"),
    "COVID crash":             ("2020-02-01", "2020-04-01"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _months_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)


def _extreme_duration(series: pd.Series, start: pd.Timestamp, func) -> int:
    """Months from start to the extreme (idxmax/idxmin) within MAX_LAG months."""
    window_end = start + pd.DateOffset(months=MAX_LAG)
    w = series[(series.index > start) & (series.index <= window_end)].dropna()
    if w.empty:
        return pd.NA
    return _months_diff(func(w), start)


def _indicator_magnitude(series: pd.Series, start: pd.Timestamp, direction: str) -> float | None:
    """Peak-to-trough magnitude within MAX_LAG months after start.
    direction='up'  → pp rise  (trough to peak, unemployment)
    direction='down'→ % drop   (peak to trough, tax receipts)
    """
    window_end = start + pd.DateOffset(months=MAX_LAG)
    w = series[(series.index > start) & (series.index <= window_end)].dropna()
    if w.empty:
        return None
    if direction == "up":
        return round(w.max() - w.min(), 2)
    else:
        peak = w.max()
        if peak <= 0:
            return None
        return round((peak - w.min()) / peak * 100, 2)


def _indicator_lag(series: pd.Series, start: pd.Timestamp, direction: str,
                   threshold: float, mom_noise_pct: float = None) -> int | None:
    # "up"   (unemployment, in pp): threshold = absolute pp rise from running trough
    # "down" (tax, in %):           threshold = relative % cumulative drop from running peak
    after = series[series.index > start].dropna()
    if after.empty:
        return None

    if direction == "up":
        for i in range(1, len(after)):
            window   = after.iloc[:i]
            prev_min = window.min()
            if after.iloc[i] - prev_min >= threshold:
                turn_date = window[window == prev_min].index[-1]
                lag = _months_diff(turn_date, start)
                return lag if lag <= MAX_LAG else None
    else:
        # Find confirmation trigger: cumul >= threshold AND mom >= mom_noise_pct.
        # Trigger must fall within MAX_LAG months to exclude false positives.
        trigger_i = None
        for i in range(1, len(after)):
            if _months_diff(after.index[i], start) > MAX_LAG:
                break
            prev_max = after.iloc[:i].max()
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

        # Scan BACKWARD from trigger to find the EARLIEST real drop.
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
        lag = _months_diff(turn_date, start)
        return lag if lag <= MAX_LAG else None

    return None

def _find_periods_fall(series: pd.Series, fall_threshold: float, recovery_threshold: float) -> list:
    """
    Falling periods using local peaks, with explicit recovery.
    fall_threshold: NEGATIVE percent (e.g. -10)
    recovery_threshold: POSITIVE percent (e.g. +5)
    """
    raw_periods = []
    in_down = False
    peak_val = series.iloc[0]
    peak_date = series.index[0]
    trough_val = peak_val
    for date, value in series.items():
        if not in_down:
            if value > peak_val:
                peak_val = value
                peak_date = date
                continue
            pct_change = (value - peak_val) / peak_val * 100
            if pct_change <= fall_threshold:
                in_down = True
                trough_val = value
                start_date = peak_date
        else:
            trough_val = min(trough_val, value)
            recovery_pct = (value - trough_val) / trough_val * 100
            if recovery_pct >= recovery_threshold:
                raw_periods.append((start_date, date))
                in_down = False
                peak_val = value
                peak_date = date
    if in_down:
        raw_periods.append((start_date, series.index[-1]))
    # --- Merge consecutive / overlapping periods ---
    if not raw_periods:
        return []
    raw_periods = sorted(raw_periods, key=lambda x: x[0])
    merged_periods = [raw_periods[0]]
    for start, end in raw_periods[1:]:
        last_start, last_end = merged_periods[-1]
        if start <= last_end:
            merged_periods[-1] = (last_start, max(last_end, end))
        else:
            merged_periods.append((start, end))
    return merged_periods

def _find_periods_unemp(series: pd.Series, rise_threshold: float, recovery_threshold: float) -> list:
    """
    Return merged (trough_date, end_date) periods where the series rises
    at least `rise_threshold` percentage points from a local trough,
    and ends after a recovery of `recovery_threshold` percentage points.
    rise_threshold      : POSITIVE (e.g. 2 for +2pp)
    recovery_threshold  : POSITIVE (e.g. 1 for -1pp recovery)
    """
    raw_periods = []
    in_up = False
    trough_val = series.iloc[0]
    trough_date = series.index[0]
    peak_val = trough_val
    for date, value in series.items():
        if not in_up:
            if value < trough_val:
                trough_val = value
                trough_date = date
                continue
            if value - trough_val >= rise_threshold:
                in_up = True
                peak_val = value
                start_date = trough_date
        else:
            peak_val = max(peak_val, value)
            if peak_val - value >= recovery_threshold:
                raw_periods.append((start_date, date))
                in_up = False
                trough_val = value
                trough_date = date
    if in_up:
        raw_periods.append((start_date, series.index[-1]))
    
    # --- Merge consecutive / overlapping periods ---
    if not raw_periods:
        return []
    raw_periods = sorted(raw_periods, key=lambda x: x[0])
    merged_periods = [raw_periods[0]]
    for start, end in raw_periods[1:]:
        last_start, last_end = merged_periods[-1]
        if start <= last_end:
            merged_periods[-1] = (last_start, max(last_end, end))
        else:
            merged_periods.append((start, end))
    return merged_periods

def _build_event_metadata(
    periods,
    series: pd.Series,
    named_events: dict,
    direction: str = "down",
):
    """
    Build (label, name, start_dt, extreme_dt, event_start) metadata.

    Parameters
    ----------
    periods : list of (start_dt, end_dt)        Detected periods (peak→recovery for down, trough→end for up)
    series : pd.Series        Time series used to locate extrema
    named_events : dict        {event_name: (start_date, end_date)}
    direction : {"down", "up"}        down → extreme = idxmin        up   → extreme = idxmax
    """

    # Parse named events once
    named_ts = {
        name: (pd.Timestamp(s), pd.Timestamp(e))
        for name, (s, e) in named_events.items()
    }
    output = []
    for start_dt, end_dt in periods:
        window = series[start_dt:end_dt]
        # --- extreme inside period ---
        if direction == "down":
            extreme_dt = window.idxmin()
        elif direction == "up":
            extreme_dt = window.idxmax()
        else:
            raise ValueError("direction must be 'down' or 'up'")
        # --- fallback label ---
        label = f"Downturn {start_dt.strftime('%b %Y')}"
        name = label
        event_start = start_dt
        # --- overlap-based named event linking ---
        best_overlap = pd.Timedelta(0)
        for ev_name, (ev_start, ev_end) in named_ts.items():
            overlap_start = max(start_dt, ev_start)
            overlap_end = min(end_dt, ev_end)
            if overlap_start <= overlap_end:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    name = ev_name
                    event_start = ev_start
        output.append(
            (name, start_dt, extreme_dt, event_start)
        )
    return output

# ── combined event builder ────────────────────────────────────────────────────

def build_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every bear-market cycle (drawdown > SP500_THRESHOLD from ATH), compute
    catalog metadata AND per-lag changes vs pre-event baseline in one pass.

    Returns long-format DataFrame: one row per event × lag (1..MAX_LAG).
    Catalog columns (label through duration_trough_tax) are repeated per lag row.
    """
    close  = df["close"].dropna()
    unemp  = df["unemployment_rate"].dropna()
    fedtax = df["receipts_bn"].dropna()

    # Find periods
    periods_sp500 = _find_periods_fall(close,  SP500_THRESHOLD,5)
    periods_unemp = _find_periods_unemp(unemp,  UNEMPLOYMENT_THRESHOLD,1)
    periods_tax   = _find_periods_fall(fedtax, FED_TAX_THRESHOLD,5)

    # Build (label, name, peak_dt, trough_date, start) list
    event_meta_sp500 = _build_event_metadata(periods_sp500,close,NAMED_EVENTS,direction="down")
    event_meta_unemp = _build_event_metadata(periods_unemp,unemp,NAMED_EVENTS,direction="up")
    event_meta_tax = _build_event_metadata(periods_tax,fedtax,NAMED_EVENTS,direction="down")

    print("\nMeta S&P500")
    for events in event_meta_sp500:
        print(events)

    print("\nMeta unemployment")
    for events in event_meta_unemp:
        print(events)

    print("\nMeta Federal tax")
    for events in event_meta_tax:
        print(events)

    # Ensure named events are included even if their drawdown didn't reach the threshold
    covered = {e[0] for e in event_meta_sp500}
    for name, (ns, ne) in NAMED_EVENTS.items():
        if name in covered:
            continue
        window = close[pd.Timestamp(ns):pd.Timestamp(ne)]
        if window.empty:
            continue
        peak_dt     = window.idxmax()
        trough_date = window.idxmin()
        event_meta_sp500.append((f"drop_{peak_dt.strftime('%Y-%m')}", name, peak_dt, trough_date, pd.Timestamp(ns)))


    print("\nEVENT_META_SP500:\n")
    for event in event_meta_sp500:
        print(event)
    print("\n")

    #---------------------
    rows = []
    for name, peak_dt, trough_date, start in event_meta_sp500:
        catalog_meta = {
            "name":                        name,
            "start_date_sp500":            start,
            "trough_date_sp500":           trough_date,
            "peak_close_sp500":            round(close[peak_dt], 2),
            "trough_close_sp500":          round(close[trough_date], 2),
            "pct_drop_sp500":              round((close[trough_date] - close[peak_dt]) / close[peak_dt] * 100, 2),
            "duration_trough_sp500":       _months_diff(trough_date, peak_dt),
            "unemp_lag_months":            (ul := _indicator_lag(df["unemployment_rate"], start, "up",   UNEMPLOYMENT_THRESHOLD)),
            "unemp_rise_pp":               _indicator_magnitude(df["unemployment_rate"], start, "up")           if ul is not None else None,
            "duration_trough_unemp":       _extreme_duration(df["unemployment_rate"], start, lambda s: s.idxmax()) if ul is not None else None,
            "tax_lag_months":              (tl := _indicator_lag(df["receipts_bn"], start, "down", FED_TAX_THRESHOLD, mom_noise_pct=MOM_NOISE_PCT)),
            "tax_drop_pct":                _indicator_magnitude(df["receipts_bn"], start, "down")               if tl is not None else None,
            "duration_trough_tax":         _extreme_duration(df["receipts_bn"], start, lambda s: s.idxmin())    if tl is not None else None,
        }
        #row = dict(catalog_meta)
        #rows.append(row)

        for lag in range(1, MAX_LAG + 1):
            target = (start + pd.DateOffset(months=lag)).to_period("M").to_timestamp()
            if target not in df.index:
                continue
            obs = df.loc[target]
            row = dict(catalog_meta)
            row["lag"]          = lag
            row["unemp_change"] = obs["unemployment_rate"] - base["unemployment_rate"]
            row["tax_change"]   = (obs["receipts_bn"] - base["receipts_bn"]) / base["receipts_bn"] * 100
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(["pct_drop_sp500"])
    #for col in ("unemp_lag_months", "tax_lag_months", "duration_trough_unemp", "duration_trough_tax"):
    #    result[col] = result[col].astype("Int64")
    print("\n\nRESULTS:\n")
    print(result)
    return result


def aggregate_event_study(raw: pd.DataFrame) -> pd.DataFrame:
    """Average changes across events per lag, with a one-sample t-test vs 0."""
    rows = []
    for lag, sub in raw.groupby("lag"):
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
    """
    sp = df["pct_change"].dropna()

    rows = []
    for lag in range(1, MAX_LAG + 1):
        future_u = df["pp_change_unrate"].shift(-lag)
        future_t = df["pct_change_receipts"].shift(-lag)

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
    print("\n" + "=" * 60)
    print("RESEARCH QUESTION ANSWER")
    print("How long after a U.S. stock market downturn do unemployment")
    print("and federal tax revenues change?")
    print("=" * 60)

    print("\n── PRIMARY: Event Study (conditional on identified downturns) ──")
    peak_u  = agg.loc[agg["avg_unemp_change"].idxmax(), "lag"]
    pos_u   = agg[agg["avg_unemp_change"] > 0].iloc[0]["lag"] if not agg[agg["avg_unemp_change"] > 0].empty else None
    sig_u   = agg[agg["p_unemp"] < 0.05]
    print(f"  Unemployment first rises   : lag {int(pos_u)} months after downturn start")
    print(f"  Unemployment peaks on avg  : lag {int(peak_u)} months")
    if sig_u.empty:
        print(f"  Statistical significance   : no lag reaches p<0.05 (only {len(agg['n_events'].unique())} events)")
    else:
        print(f"  Significant lags (p<0.05)  : {sig_u['lag'].tolist()}")

    neg_t = agg[agg["avg_tax_change"] < 0]
    if not neg_t.empty:
        print(f"  Tax receipts first fall    : lag {int(neg_t.iloc[0]['lag'])} months")
    else:
        print(f"  Tax receipts               : avg stays above pre-event baseline across all lags")
        print(f"                               (long-term upward trend dominates; see per-event catalog)")

    print("\n  Aggregated changes by lag:")
    print(agg[["lag", "avg_unemp_change", "avg_tax_change", "p_unemp", "p_tax"]].to_string(index=False))

    print("\n── SUPPORTING: Cross-Correlation (all months, not event-conditional) ──")
    print("  Note: uses the full time series (~900 months), not just the downturn events.")
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
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    print(f"  {len(df)} rows  ({df.index.min().date()} → {df.index.max().date()})")

    print("Building events...")
    events   = build_events(df)
    n_events = events["name"].nunique()
    print(f"  {n_events} events (drawdown threshold {SP500_THRESHOLD}%):")
    for _, r in events.drop_duplicates("name").sort_values("start_date").iterrows():
        print(f"    {r['name']}  →  start {pd.Timestamp(r['start_date']).date()}")
    events.to_csv(EVENTS_OUT, index=False)
    print(f"  Saved {EVENTS_OUT.name}  ({n_events} events, {len(events)} rows)")

    print("Running aggregation...")
    agg   = aggregate_event_study(events)

    print("Running cross-correlation analysis...")
    xcorr = run_cross_correlation(df)

    results = agg.merge(xcorr, on="lag", suffixes=("_event", "_xcorr"))
    results.to_csv(LAG_OUT, index=False)
    print(f"  Saved {LAG_OUT.name}")

    print_summary(agg, xcorr)
    return df, results, events


if __name__ == "__main__":
    run()
