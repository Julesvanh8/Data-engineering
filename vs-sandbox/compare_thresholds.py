"""
compare_thresholds.py — compare unemp/tax lag results for different thresholds.

Unemployment uses absolute pp (percentage points).
Tax uses relative % (since it is in bn$, not a rate).

Usage:
    python vs-sandbox/compare_thresholds.py
    python vs-sandbox/compare_thresholds.py --debug "Downturn Dec 1968"
"""

import pandas as pd
from pathlib import Path

CATALOG_CSV       = Path(__file__).resolve().parents[1] / "data" / "processed" / "downturn_catalog.csv"
DATA_PATH         = Path(__file__).resolve().parents[1] / "data" / "processed" / "merged_monthly_vs.csv"

UNEMP_THRESHOLDS_PP = [2.5, 3.0, 3.5, 4.0]
TAX_THRESHOLDS_PCT  = [7.5]
TAX_MOM_NOISE_PCT   = 0.5
MAX_LAG             = 24


def indicator_lag(series: pd.Series, start: pd.Timestamp, direction: str,
                  confirm: float, mom_noise_pct: float = None,
                  debug: bool = False) -> int | None:
    after = series[series.index > start].dropna()
    if after.empty:
        return None

    if direction == "up":
        for i in range(1, len(after)):
            window   = after.iloc[:i]
            prev_min = window.min()
            move     = after.iloc[i] - prev_min
            if move >= confirm:
                turn_date = window[window == prev_min].index[-1]
                lag = (turn_date.year - start.year) * 12 + (turn_date.month - start.month)
                return lag if lag <= MAX_LAG else None

    else:  # tax / "down"
        # Find confirmation trigger: cumul >= confirm AND mom >= mom_noise_pct.
        # Trigger must fall within MAX_LAG months.
        trigger_i = None
        for i in range(1, len(after)):
            trigger_lag = (after.index[i].year - start.year) * 12 + (after.index[i].month - start.month)
            if trigger_lag > MAX_LAG:
                break
            window   = after.iloc[:i]
            prev_max = window.max()
            if prev_max <= 0:
                continue
            cumul_drop = (prev_max - after.iloc[i]) / prev_max * 100
            prev_val   = after.iloc[i - 1]
            mom_drop   = (prev_val - after.iloc[i]) / prev_val * 100 if prev_val > 0 else 0

            if debug:
                print(f"  {after.index[i].date()}  val={after.iloc[i]:.2f}  "
                      f"peak={prev_max:.2f}  cumul={cumul_drop:.2f}%  mom={mom_drop:.2f}%  "
                      f"{'CUMUL OK' if cumul_drop >= confirm else '':10}"
                      f"{'MOM OK' if mom_drop >= mom_noise_pct else 'mom too small'}")

            if cumul_drop >= confirm:
                if mom_noise_pct is not None and mom_drop < mom_noise_pct:
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


def main():
    catalog = pd.read_csv(CATALOG_CSV, parse_dates=["start_date"])
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()

    events = catalog.sort_values("start_date")[["name", "start_date"]].values

    # ── debug: step-by-step tax detection for 1968 ───────────────────────────
    row1968 = catalog[catalog["name"].str.contains("1968")]
    if not row1968.empty:
        start68 = pd.Timestamp(row1968.iloc[0]["start_date"])
        after   = df["receipts_bn"][df.index > start68].dropna()
        print(f"\nDEBUG: Federal tax step-by-step for 'Downturn Dec 1968' (start {start68.date()})")
        print(f"  threshold={TAX_THRESHOLDS_PCT[0]}%  mom_noise={TAX_MOM_NOISE_PCT}%")
        print(f"  {'Date':<12} {'Value':>8} {'RunPeak':>8} {'Cumul%':>8} {'MoM%':>8}  Status")
        print(f"  {'-'*60}")
        running_max = None
        prev_val    = None
        for date, val in after.items():
            if running_max is None or val > running_max:
                running_max = val
            cumul = (running_max - val) / running_max * 100 if running_max > 0 else 0
            mom   = (prev_val - val) / prev_val * 100 if prev_val and prev_val > 0 else 0
            cumul_ok = cumul >= TAX_THRESHOLDS_PCT[0]
            mom_ok   = mom   >= TAX_MOM_NOISE_PCT
            status = ""
            if cumul_ok and mom_ok:
                status = "<<< TRIGGER"
            elif cumul_ok:
                status = "cumul ok, mom too small"
            print(f"  {str(date.date()):<12} {val:>8.2f} {running_max:>8.2f} {cumul:>8.2f} {mom:>8.2f}  {status}")
            if cumul_ok and mom_ok:
                break
            prev_val = val
        print()

    # ── debug: step-by-step tax detection for 1987 ───────────────────────────
    row1987 = catalog[catalog["name"].str.contains("1987")]
    if not row1987.empty:
        start87 = pd.Timestamp(row1987.iloc[0]["start_date"])
        after   = df["receipts_bn"][df.index > start87].dropna()
        print(f"\nDEBUG: Federal tax step-by-step for 'Downturn Aug 1987' (start {start87.date()})")
        print(f"  threshold={TAX_THRESHOLDS_PCT[0]}%  mom_noise={TAX_MOM_NOISE_PCT}%")
        print(f"  {'Date':<12} {'Value':>8} {'RunPeak':>8} {'Cumul%':>8} {'MoM%':>8}  Status")
        print(f"  {'-'*60}")
        running_max = None
        prev_val    = None
        for date, val in after.items():
            if running_max is None or val > running_max:
                running_max = val
            cumul = (running_max - val) / running_max * 100 if running_max > 0 else 0
            mom   = (prev_val - val) / prev_val * 100 if prev_val and prev_val > 0 else 0
            cumul_ok = cumul >= TAX_THRESHOLDS_PCT[0]
            mom_ok   = mom   >= TAX_MOM_NOISE_PCT
            status = ""
            if cumul_ok and mom_ok:
                status = "<<< TRIGGER"
            elif cumul_ok:
                status = "cumul ok, mom too small"
            print(f"  {str(date.date()):<12} {val:>8.2f} {running_max:>8.2f} {cumul:>8.2f} {mom:>8.2f}  {status}")
            if cumul_ok and mom_ok:
                break
            prev_val = val
        print()

    # ── unemployment: pp thresholds ───────────────────────────────────────────
    print("\nUNEMPLOYMENT LAG (pp — absolute pp rise from trough, last month at trough)")
    header = f"  {'Event':<32}" + "".join(f"  {p}pp".rjust(8) for p in UNEMP_THRESHOLDS_PP)
    print(header)
    print("  " + "-" * (32 + 8 * len(UNEMP_THRESHOLDS_PP)))
    for name, start in events:
        start = pd.Timestamp(start)
        row   = f"  {name:<32}"
        for p in UNEMP_THRESHOLDS_PP:
            u = indicator_lag(df["unemployment_rate"], start, "up", p)
            row += f"  {str(u) if u is not None else '-':>6}"
        print(row)

    # ── tax: % thresholds ─────────────────────────────────────────────────────
    print(f"\nFEDERAL TAX LAG (% cumulative drop from peak, mom noise filter={TAX_MOM_NOISE_PCT}%)")
    header = f"  {'Event':<32}" + "".join(f"  {p}%".rjust(8) for p in TAX_THRESHOLDS_PCT)
    print(header)
    print("  " + "-" * (32 + 8 * len(TAX_THRESHOLDS_PCT)))
    for name, start in events:
        start = pd.Timestamp(start)
        row   = f"  {name:<32}"
        for p in TAX_THRESHOLDS_PCT:
            t = indicator_lag(df["receipts_bn"], start, "down", p, TAX_MOM_NOISE_PCT)
            row += f"  {str(t) if t is not None else '-':>6}"
        print(row)


if __name__ == "__main__":
    main()
