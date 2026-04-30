from pathlib import Path
import pandas as pd


# =========================
# 1. File paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

MAIN_EVENTS = PROCESSED / "events_combined.csv"
ROBUST_EVENTS = PROCESSED / "events_combined_robust.csv"

OUTPUT_FILE = PROCESSED / "robustness_summary.csv"


IMPORTANT_EVENTS = [
    "Dot-com crash",
    "Global Financial Crisis",
    "COVID crash",
]


# =========================
# 2. Load one row per event
# =========================

def load_event_summary(path: Path, version: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=[
            "sp500_start",
            "sp500_trough",
            "unemp_event_start",
            "unemp_peak_date",
            "tax_event_start",
            "tax_trough_date",
        ],
    )

    keep_columns = [
        "name",
        "sp500_start",
        "sp500_trough",
        "sp500_pct_drop",
        "unemp_lag_months",
        "tax_lag_months",
    ]

    df = (
        df[keep_columns]
        .drop_duplicates(subset=["name"])
        .sort_values("sp500_start")
        .reset_index(drop=True)
    )

    df = df.rename(
        columns={
            "sp500_start": f"sp500_start_{version}",
            "sp500_trough": f"sp500_trough_{version}",
            "sp500_pct_drop": f"sp500_pct_drop_{version}",
            "unemp_lag_months": f"unemp_lag_{version}",
            "tax_lag_months": f"tax_lag_{version}",
        }
    )

    return df


# =========================
# 3. Compare nominal vs real
# =========================

def compare_robustness() -> pd.DataFrame:
    if not MAIN_EVENTS.exists():
        raise FileNotFoundError(f"Missing file: {MAIN_EVENTS}")

    if not ROBUST_EVENTS.exists():
        raise FileNotFoundError(f"Missing file: {ROBUST_EVENTS}")

    nominal = load_event_summary(MAIN_EVENTS, "nominal")
    real = load_event_summary(ROBUST_EVENTS, "real")

    comparison = nominal.merge(
        real,
        on="name",
        how="outer",
        indicator=True,
    )

    comparison["detected_in_both"] = comparison["_merge"] == "both"

    comparison["unemp_lag_difference"] = (
        comparison["unemp_lag_real"] - comparison["unemp_lag_nominal"]
    )

    comparison["tax_lag_difference"] = (
        comparison["tax_lag_real"] - comparison["tax_lag_nominal"]
    )

    comparison["drop_difference"] = (
        comparison["sp500_pct_drop_real"] - comparison["sp500_pct_drop_nominal"]
    )

    comparison = comparison.drop(columns=["_merge"])

    comparison.to_csv(OUTPUT_FILE, index=False)

    return comparison


# =========================
# 4. Print readable conclusion
# =========================

def print_readable_summary(comparison: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("ROBUSTNESS CHECK: NOMINAL S&P 500 VS REAL S&P 500")
    print("=" * 80)

    important = comparison[comparison["name"].isin(IMPORTANT_EVENTS)].copy()

    print("\n1. Important crash events")
    print("-" * 80)

    for event in IMPORTANT_EVENTS:
        row = important[important["name"] == event]

        if row.empty:
            print(f"{event}: not found in the comparison file")
            continue

        row = row.iloc[0]

        status = "YES" if row["detected_in_both"] else "NO"

        print(f"{event}")
        print(f"  Detected in both nominal and real analysis: {status}")

        if row["detected_in_both"]:
            print(f"  Nominal S&P drop: {row['sp500_pct_drop_nominal']:.2f}%")
            print(f"  Real S&P drop:    {row['sp500_pct_drop_real']:.2f}%")

            if pd.notna(row["unemp_lag_nominal"]) and pd.notna(row["unemp_lag_real"]):
                print(
                    f"  Unemployment lag: nominal {int(row['unemp_lag_nominal'])} months, "
                    f"real {int(row['unemp_lag_real'])} months"
                )
            else:
                print("  Unemployment lag: not detected in one of the versions")

            if pd.notna(row["tax_lag_nominal"]) and pd.notna(row["tax_lag_real"]):
                print(
                    f"  Tax lag: nominal {int(row['tax_lag_nominal'])} months, "
                    f"real {int(row['tax_lag_real'])} months"
                )
            else:
                print("  Tax lag: not detected in one of the versions")

        print()

    print("2. Overall robustness interpretation")
    print("-" * 80)

    important_detected = important["detected_in_both"].sum()
    total_important = len(IMPORTANT_EVENTS)

    both_all = comparison[comparison["detected_in_both"]]
    nominal_only = comparison[~comparison["detected_in_both"] & comparison["sp500_start_nominal"].notna()]
    real_only = comparison[~comparison["detected_in_both"] & comparison["sp500_start_real"].notna()]

    print(f"Important events detected in both versions: {important_detected}/{total_important}")
    print(f"Total events detected in both versions: {len(both_all)}")
    print(f"Events only in nominal version: {len(nominal_only)}")
    print(f"Events only in real version: {len(real_only)}")

    print("\nConclusion:")

    if important_detected == total_important:
        print(
            "The robustness check supports the main analysis. "
            "The three main historical downturns are detected both with nominal prices "
            "and with inflation-adjusted real prices. This means the main event detection "
            "is not only driven by nominal price movements."
        )
    else:
        print(
            "The robustness check is mixed. Some important events are not detected in both versions. "
            "This means the results are partly sensitive to whether nominal or real S&P 500 prices are used."
        )

    print("\nSaved summary file:")
    print(OUTPUT_FILE)
    print("=" * 80 + "\n")


# =========================
# 5. Main
# =========================

def main() -> None:
    comparison = compare_robustness()
    print_readable_summary(comparison)


if __name__ == "__main__":
    main()