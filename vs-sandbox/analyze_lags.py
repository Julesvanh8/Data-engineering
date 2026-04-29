"""
analyze_lags.py — event-study lag analysis for the pipeline.

Research question:
    How long after a U.S. stock market downturn do unemployment and
    federal income tax revenues change?

Approach:
    Event study   — track average change vs pre-downturn baseline
                    at each lag (1-MAX_LAG months) across all detected events.

Outputs:
    data/processed/events_combined.csv  (per-event, per-lag raw changes + catalog metadata)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH  = Path(__file__).resolve().parents[1] / "data" / "processed" / "merged_monthly_vs.csv"
EVENTS_OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "events_combined.csv"

MAX_LAG                 = 23
SP500_THRESHOLD         = -19.0  # % fall from running all-time high to trigger a bear-market event
UNEMPLOYMENT_THRESHOLD  =  2     # pp absolute rise from trough to confirm unemployment increase
FED_TAX_THRESHOLD       = -7.5   # % cumulative drop from peak to confirm tax revenue decline
MOM_NOISE_PCT           =  1.5   # % min month-to-month drop to count as a real tax decline move
FED_TAX_PCT_MARGIN      =  1.6   # % margin to consider consecutive peaks as flat (for tax revenue)
BASELINE_MONTHS         =  6     # months before event start used as pre-event baseline

NAMED_EVENTS = {
    "Dot-com crash":           ("2000-03-01", "2002-10-01"),
    "Global Financial Crisis": ("2007-10-01", "2009-03-01"),
    "COVID crash":             ("2020-02-01", "2020-04-01"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _months_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)

def _find_periods_fall(series: pd.Series, fall_threshold: float, recovery_threshold: float, pct_margin: float = 0.0) -> list:
    """
    Falling periods using local peaks, with explicit recovery.

    If multiple consecutive peak values are equal or within `pct_margin` percent,
    the START date is set to the LAST such occurrence.

    fall_threshold     : NEGATIVE percent (e.g. -10)
    recovery_threshold : POSITIVE percent (e.g. +5)
    pct_margin         : NON-NEGATIVE percent (e.g. 0.5 for ±0.5% flat peak)
    """

    raw_periods = []
    in_down = False

    peak_val = series.iloc[0]
    peak_date = series.index[0]

    trough_val = peak_val

    for date, value in series.items():

        if not in_down:
            # New higher peak
            if value > peak_val:
                peak_val = value
                peak_date = date
                continue

            # Flat peak within pct_margin → shift peak forward
            peak_pct_diff = abs(value - peak_val) / peak_val * 100
            if peak_pct_diff <= pct_margin:
                peak_date = date
                continue

            # Fall detected
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

def _find_periods_unemp(series: pd.Series, rise_threshold: float, recovery_threshold: float, pp_margin: float = 0.0, confirm_months: int = 3) -> list:
    """
    Unemployment rise periods with confirmation requirement.
    """

    raw_periods = []
    in_up = False
    confirm_count = 0

    trough_val = series.iloc[0]
    trough_date = series.index[0]
    peak_val = trough_val

    for date, value in series.items():
        if not in_up:
            # update trough (flat logic preserved)
            if value < trough_val:
                trough_val = value
                trough_date = date
                confirm_count = 0
                continue

            if abs(value - trough_val) <= pp_margin:
                trough_date = date
                confirm_count = 0
                continue

            # tentative rise
            if value - trough_val >= rise_threshold:
                confirm_count += 1
                if confirm_count >= confirm_months:
                    in_up = True
                    peak_val = value
                    start_date = trough_date
            else:
                confirm_count = 0

        else:
            peak_val = max(peak_val, value)

            if peak_val - value >= recovery_threshold:
                raw_periods.append((start_date, date))
                in_up = False
                trough_val = value
                trough_date = date
                confirm_count = 0

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

def _first_rise_period(series: pd.Series, start: pd.Timestamp,
                       rise_threshold: float, recovery_threshold: float,
                       confirm_months: int) -> tuple:
    """Re-scan from `start`, return (event_start, peak_dt, lag) of first detected rise."""
    sub = series[series.index >= start]
    periods = _find_periods_unemp(sub, rise_threshold, recovery_threshold,
                                   confirm_months=confirm_months)
    if not periods:
        return None, None, None
    p_start, p_end = periods[0]
    peak_dt = sub[p_start:p_end].idxmax()
    lag = _months_diff(p_start, start)
    return (None, None, None) if lag >= MAX_LAG else (p_start, peak_dt, lag)


def _first_fall_period(series: pd.Series, start: pd.Timestamp,
                       fall_threshold: float, recovery_threshold: float,
                       pct_margin: float) -> tuple:
    """Re-scan from `start`, return (event_start, trough_dt, lag) of first detected fall."""
    sub = series[series.index >= start]
    periods = _find_periods_fall(sub, fall_threshold, recovery_threshold, pct_margin)
    if not periods:
        return None, None, None
    p_start, p_end = periods[0]
    trough_dt = sub[p_start:p_end].idxmin()
    lag = _months_diff(p_start, start)
    return (None, None, None) if lag >= MAX_LAG else (p_start, trough_dt, lag)


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

    periods_sp500    = _find_periods_fall(close, SP500_THRESHOLD, 5)
    event_meta_sp500 = _build_event_metadata(periods_sp500, close, NAMED_EVENTS, direction="down")

    # Ensure named events are included even if drawdown didn't reach the threshold
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

    rows = []
    for sp_name, sp_peak, sp_trough, sp_start in event_meta_sp500:
        sp500_peak_val   = close[sp_peak]
        sp500_trough_val = close[sp_trough]

        unemp_start, unemp_peak, unemp_lag = _first_rise_period(
            unemp, sp_start, UNEMPLOYMENT_THRESHOLD, 1, confirm_months=3)
        tax_start, tax_trough, tax_lag = _first_fall_period(
            fedtax, sp_start, FED_TAX_THRESHOLD, 5, FED_TAX_PCT_MARGIN)

        catalog_meta = {
            "name":              sp_name,
            "sp500_start":       sp_start,
            "sp500_trough":      sp_trough,
            "sp500_pct_drop":    round((sp500_trough_val - sp500_peak_val) / sp500_peak_val * 100, 2),
            "sp500_duration":    _months_diff(sp_trough, sp_peak),
            "unemp_event_start": unemp_start,
            "unemp_peak_date":   unemp_peak,
            "unemp_lag_months":  unemp_lag,
            "tax_event_start":   tax_start,
            "tax_trough_date":   tax_trough,
            "tax_lag_months":    tax_lag,
        }

        # Pre-event baseline for event study
        base_w = df[(df.index >= sp_start - pd.DateOffset(months=BASELINE_MONTHS))
                    & (df.index < sp_start)]
        base_u = base_w["unemployment_rate"].mean()
        base_t = base_w["receipts_bn"].mean()

        for lag in range(1, MAX_LAG + 1):
            target = (sp_start + pd.DateOffset(months=lag)).to_period("M").to_timestamp()
            if target not in df.index:
                continue
            obs = df.loc[target]
            row = dict(catalog_meta)
            row["lag"]          = lag
            row["unemp_change"] = obs["unemployment_rate"] - base_u if pd.notna(base_u) else None
            row["tax_change"]   = (obs["receipts_bn"] - base_t) / base_t * 100 \
                                  if (pd.notna(base_t) and base_t > 0) else None
            rows.append(row)

    return pd.DataFrame(rows)


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


# ── summary print ─────────────────────────────────────────────────────────────

def print_summary(agg: pd.DataFrame) -> None:
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

    print("=" * 60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def run(data_path=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(data_path) if data_path else DATA_PATH
    print("Loading data...")
    if path.suffix == ".parquet" or path.is_dir():
        df = pd.read_parquet(path).set_index("date")
    else:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    print(f"  {len(df)} rows  ({df.index.min().date()} → {df.index.max().date()})")

    print("Building events...")
    events = build_events(df)

    if events.empty:
        raise RuntimeError(
            "No events were detected. Check the S&P 500 threshold or the input data."
        )

    n_events = events["name"].nunique()
    print(f"  {n_events} events (drawdown threshold {SP500_THRESHOLD}%):")

    for _, r in events.drop_duplicates("name").sort_values("sp500_start").iterrows():
        print(f"    {r['name']}  →  start {pd.Timestamp(r['sp500_start']).date()}")

    events.to_csv(EVENTS_OUT, index=False)
    print(f"  Saved {EVENTS_OUT.name}  ({n_events} events, {len(events)} rows)")

    print("Running aggregation...")
    agg = aggregate_event_study(events)

    print_summary(agg)

    return df, events


if __name__ == "__main__":
    run()