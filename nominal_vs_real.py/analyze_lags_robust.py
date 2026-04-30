"""
analyze_lags_robust.py — robustness check for the event-study lag analysis.

This script repeats Valerie's lag analysis using inflation-adjusted
S&P 500 real prices instead of nominal S&P 500 prices.

Inputs:
    data/processed/merged_monthly_vs_robust.csv

Outputs:
    data/processed/lag_results_robust.csv
    data/processed/events_combined_robust.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "merged_monthly_vs_robust.csv"
)

LAG_OUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "lag_results_robust.csv"
)

EVENTS_OUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "events_combined_robust.csv"
)


MAX_LAG = 23
SP500_THRESHOLD = -19.0
UNEMPLOYMENT_THRESHOLD = 2
FED_TAX_THRESHOLD = -7.5
FED_TAX_PCT_MARGIN = 1.6
BASELINE_MONTHS = 6


NAMED_EVENTS = {
    "Dot-com crash": ("2000-03-01", "2002-10-01"),
    "Global Financial Crisis": ("2007-10-01", "2009-03-01"),
    "COVID crash": ("2020-02-01", "2020-04-01"),
}


# =========================
# Helper functions
# =========================

def _months_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)


def _find_periods_fall(
    series: pd.Series,
    fall_threshold: float,
    recovery_threshold: float,
    pct_margin: float = 0.0,
) -> list:
    """
    Find falling periods using local peaks and explicit recovery.

    This robust version removes zero or negative values before calculating
    percentage changes, because percentage changes cannot be calculated
    with a zero denominator.
    """

    series = series.dropna()
    series = series[series > 0]

    if series.empty:
        return []

    raw_periods = []
    in_down = False

    peak_val = series.iloc[0]
    peak_date = series.index[0]
    trough_val = peak_val

    for date, value in series.items():

        if value <= 0:
            continue

        if not in_down:
            if value > peak_val:
                peak_val = value
                peak_date = date
                continue

            if peak_val <= 0:
                continue

            peak_pct_diff = abs(value - peak_val) / peak_val * 100

            if peak_pct_diff <= pct_margin:
                peak_date = date
                continue

            pct_change = (value - peak_val) / peak_val * 100

            if pct_change <= fall_threshold:
                in_down = True
                trough_val = value
                start_date = peak_date

        else:
            trough_val = min(trough_val, value)

            if trough_val <= 0:
                continue

            recovery_pct = (value - trough_val) / trough_val * 100

            if recovery_pct >= recovery_threshold:
                raw_periods.append((start_date, date))
                in_down = False
                peak_val = value
                peak_date = date
                trough_val = value

    if in_down:
        raw_periods.append((start_date, series.index[-1]))

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


def _find_periods_unemp(
    series: pd.Series,
    rise_threshold: float,
    recovery_threshold: float,
    pp_margin: float = 0.0,
    confirm_months: int = 3,
) -> list:
    """
    Find unemployment rise periods with confirmation requirement.
    """

    series = series.dropna()

    if series.empty:
        return []

    raw_periods = []
    in_up = False
    confirm_count = 0

    trough_val = series.iloc[0]
    trough_date = series.index[0]
    peak_val = trough_val

    for date, value in series.items():

        if not in_up:
            if value < trough_val:
                trough_val = value
                trough_date = date
                confirm_count = 0
                continue

            if abs(value - trough_val) <= pp_margin:
                trough_date = date
                confirm_count = 0
                continue

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
    Build event metadata.

    Output per event:
    name, start date, extreme date, event start date.
    """

    named_ts = {
        name: (pd.Timestamp(s), pd.Timestamp(e))
        for name, (s, e) in named_events.items()
    }

    output = []

    for start_dt, end_dt in periods:
        window = series[start_dt:end_dt].dropna()
        window = window[window > 0]

        if window.empty:
            continue

        if direction == "down":
            extreme_dt = window.idxmin()
        elif direction == "up":
            extreme_dt = window.idxmax()
        else:
            raise ValueError("direction must be 'down' or 'up'")

        label = f"Downturn {start_dt.strftime('%b %Y')}"
        name = label
        event_start = start_dt

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

        output.append((name, start_dt, extreme_dt, event_start))

    return output


def _first_rise_period(
    series: pd.Series,
    start: pd.Timestamp,
    rise_threshold: float,
    recovery_threshold: float,
    confirm_months: int,
) -> tuple:
    """
    Re-scan from start and return first unemployment rise period.
    """

    sub = series[series.index >= start].dropna()

    periods = _find_periods_unemp(
        sub,
        rise_threshold,
        recovery_threshold,
        confirm_months=confirm_months,
    )

    if not periods:
        return None, None, None

    p_start, p_end = periods[0]
    peak_dt = sub[p_start:p_end].idxmax()
    lag = _months_diff(p_start, start)

    if lag >= MAX_LAG:
        return None, None, None

    return p_start, peak_dt, lag


def _first_fall_period(
    series: pd.Series,
    start: pd.Timestamp,
    fall_threshold: float,
    recovery_threshold: float,
    pct_margin: float,
) -> tuple:
    """
    Re-scan from start and return first tax fall period.
    """

    sub = series[series.index >= start].dropna()
    sub = sub[sub > 0]

    periods = _find_periods_fall(
        sub,
        fall_threshold,
        recovery_threshold,
        pct_margin,
    )

    if not periods:
        return None, None, None

    p_start, p_end = periods[0]
    trough_dt = sub[p_start:p_end].idxmin()
    lag = _months_diff(p_start, start)

    if lag >= MAX_LAG:
        return None, None, None

    return p_start, trough_dt, lag


# =========================
# Event study
# =========================

def build_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build event-study rows using real_close instead of nominal close.
    """

    close = df["real_close"].dropna()
    close = close[close > 0]

    unemp = df["unemployment_rate"].dropna()

    fedtax = df["receipts_bn"].dropna()
    fedtax = fedtax[fedtax > 0]

    periods_sp500 = _find_periods_fall(
        close,
        SP500_THRESHOLD,
        5,
    )

    event_meta_sp500 = _build_event_metadata(
        periods_sp500,
        close,
        NAMED_EVENTS,
        direction="down",
    )

    covered = {event[0] for event in event_meta_sp500}

    for name, (ns, ne) in NAMED_EVENTS.items():
        if name in covered:
            continue

        window = close[pd.Timestamp(ns):pd.Timestamp(ne)]
        window = window[window > 0]

        if window.empty:
            continue

        peak_dt = window.idxmax()
        trough_date = window.idxmin()

        event_meta_sp500.append(
            (
                name,
                peak_dt,
                trough_date,
                pd.Timestamp(ns),
            )
        )

    rows = []

    for sp_name, sp_peak, sp_trough, sp_start in event_meta_sp500:
        sp500_peak_val = close[sp_peak]
        sp500_trough_val = close[sp_trough]

        if sp500_peak_val <= 0:
            continue

        unemp_start, unemp_peak, unemp_lag = _first_rise_period(
            unemp,
            sp_start,
            UNEMPLOYMENT_THRESHOLD,
            1,
            confirm_months=3,
        )

        tax_start, tax_trough, tax_lag = _first_fall_period(
            fedtax,
            sp_start,
            FED_TAX_THRESHOLD,
            5,
            FED_TAX_PCT_MARGIN,
        )

        catalog_meta = {
            "name": sp_name,
            "sp500_start": sp_start,
            "sp500_trough": sp_trough,
            "sp500_pct_drop": round(
                (sp500_trough_val - sp500_peak_val) / sp500_peak_val * 100,
                2,
            ),
            "sp500_duration": _months_diff(sp_trough, sp_peak),
            "unemp_event_start": unemp_start,
            "unemp_peak_date": unemp_peak,
            "unemp_lag_months": unemp_lag,
            "tax_event_start": tax_start,
            "tax_trough_date": tax_trough,
            "tax_lag_months": tax_lag,
            "stock_price_used": "real_close",
        }

        base_w = df[
            (df.index >= sp_start - pd.DateOffset(months=BASELINE_MONTHS))
            & (df.index < sp_start)
        ]

        base_u = base_w["unemployment_rate"].mean()
        base_t = base_w["receipts_bn"].mean()

        for lag in range(1, MAX_LAG + 1):
            target = (
                sp_start + pd.DateOffset(months=lag)
            ).to_period("M").to_timestamp()

            if target not in df.index:
                continue

            obs = df.loc[target]
            row = dict(catalog_meta)

            row["lag"] = lag

            row["unemp_change"] = (
                obs["unemployment_rate"] - base_u
                if pd.notna(base_u)
                else None
            )

            row["tax_change"] = (
                (obs["receipts_bn"] - base_t) / base_t * 100
                if pd.notna(base_t) and base_t > 0
                else None
            )

            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_event_study(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Average changes across events per lag, with a one-sample t-test vs 0.
    """

    rows = []

    for lag, sub in raw.groupby("lag"):
        u = sub["unemp_change"].dropna()
        t = sub["tax_change"].dropna()

        _, p_u = (
            stats.ttest_1samp(u, 0)
            if len(u) > 1
            else (np.nan, np.nan)
        )

        _, p_t = (
            stats.ttest_1samp(t, 0)
            if len(t) > 1
            else (np.nan, np.nan)
        )

        rows.append(
            {
                "lag": lag,
                "avg_unemp_change": u.mean(),
                "std_unemp_change": u.std(),
                "avg_tax_change": t.mean(),
                "std_tax_change": t.std(),
                "n_events": len(u),
                "p_unemp": round(p_u, 4),
                "p_tax": round(p_t, 4),
            }
        )

    return pd.DataFrame(rows)


# =========================
# Cross-correlation
# =========================

def run_cross_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation between real S&P 500 monthly return and future changes
    in unemployment and tax receipts.
    """

    sp = df["real_pct_change"].dropna()

    rows = []

    for lag in range(1, MAX_LAG + 1):
        future_u = df["pp_change_unrate"].shift(-lag)
        future_t = df["pct_change_receipts"].shift(-lag)

        aligned_u = pd.concat([sp, future_u], axis=1, sort=False).dropna()
        aligned_t = pd.concat([sp, future_t], axis=1, sort=False).dropna()

        r_u, p_u = (
            stats.pearsonr(aligned_u.iloc[:, 0], aligned_u.iloc[:, 1])
            if len(aligned_u) > 2
            else (np.nan, np.nan)
        )

        r_t, p_t = (
            stats.pearsonr(aligned_t.iloc[:, 0], aligned_t.iloc[:, 1])
            if len(aligned_t) > 2
            else (np.nan, np.nan)
        )

        rows.append(
            {
                "lag": lag,
                "r_unemp": round(r_u, 4),
                "p_unemp": round(p_u, 4),
                "r_tax": round(r_t, 4),
                "p_tax": round(p_t, 4),
            }
        )

    return pd.DataFrame(rows)


# =========================
# Summary
# =========================

def print_summary(agg: pd.DataFrame, xcorr: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK")
    print("Lag analysis using inflation-adjusted S&P 500 real prices")
    print("=" * 70)

    print("\nPrimary method: Event Study")

    peak_u = agg.loc[agg["avg_unemp_change"].idxmax(), "lag"]

    pos_u = (
        agg[agg["avg_unemp_change"] > 0].iloc[0]["lag"]
        if not agg[agg["avg_unemp_change"] > 0].empty
        else None
    )

    sig_u = agg[agg["p_unemp"] < 0.05]

    if pos_u is not None:
        print(f"  Unemployment first rises: lag {int(pos_u)} months")
    else:
        print("  Unemployment first rises: not detected")

    print(f"  Unemployment peaks on average: lag {int(peak_u)} months")

    if sig_u.empty:
        print("  Statistical significance: no unemployment lag reaches p < 0.05")
    else:
        print(f"  Significant unemployment lags: {sig_u['lag'].tolist()}")

    neg_t = agg[agg["avg_tax_change"] < 0]

    if not neg_t.empty:
        print(f"  Tax receipts first fall: lag {int(neg_t.iloc[0]['lag'])} months")
    else:
        print("  Tax receipts: average stays above pre-event baseline across all lags")

    print("\nAggregated event-study changes by lag:")
    print(
        agg[
            [
                "lag",
                "avg_unemp_change",
                "avg_tax_change",
                "p_unemp",
                "p_tax",
            ]
        ].to_string(index=False)
    )

    print("\nSupporting method: Cross-Correlation")

    peak_xu = xcorr.loc[xcorr["r_unemp"].abs().idxmax(), "lag"]
    peak_xt = xcorr.loc[xcorr["r_tax"].abs().idxmax(), "lag"]

    r_xu = xcorr.loc[xcorr["lag"] == peak_xu, "r_unemp"].values[0]
    r_xt = xcorr.loc[xcorr["lag"] == peak_xt, "r_tax"].values[0]

    print(f"  Strongest unemployment correlation: lag {peak_xu} months, r = {r_xu}")
    print(f"  Strongest tax correlation: lag {peak_xt} months, r = {r_xt}")

    print("=" * 70 + "\n")


# =========================
# Main
# =========================

def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading robustness dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {DATA_PATH}. "
            "Run build_dataset_robust.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()

    required_columns = [
        "real_close",
        "real_pct_change",
        "unemployment_rate",
        "receipts_bn",
        "pp_change_unrate",
        "pct_change_receipts",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in robustness dataset: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"  Rows: {len(df)}")
    print(f"  Period: {df.index.min().date()} → {df.index.max().date()}")

    print("\nBuilding robustness events with real_close...")
    events = build_events(df)

    if events.empty:
        raise RuntimeError(
            "No events were detected. Check real_close, the S&P 500 threshold or the input data."
        )

    n_events = events["name"].nunique()

    print(f"  {n_events} events detected with real price drawdown threshold {SP500_THRESHOLD}%:")

    for _, row in events.drop_duplicates("name").sort_values("sp500_start").iterrows():
        print(
            f"    {row['name']} → start {pd.Timestamp(row['sp500_start']).date()}"
        )

    events.to_csv(EVENTS_OUT, index=False)

    print(f"  Saved {EVENTS_OUT.name} with {n_events} events and {len(events)} rows")

    print("\nRunning event-study aggregation...")
    agg = aggregate_event_study(events)

    print("Running cross-correlation analysis...")
    xcorr = run_cross_correlation(df)

    results = agg.merge(
        xcorr,
        on="lag",
        suffixes=("_event", "_xcorr"),
    )

    results.to_csv(LAG_OUT, index=False)

    print(f"  Saved {LAG_OUT.name}")

    print_summary(agg, xcorr)

    return df, results, events


if __name__ == "__main__":
    run()