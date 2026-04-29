from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, date_trunc, last
from pyspark.sql.window import Window


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SP500_FILE = RAW_DIR / "github_sp500_daily.csv"
UNRATE_FILE = RAW_DIR / "fred_unrate.csv"
TAX_FILE = RAW_DIR / "fred_w006rc1q027sbea.csv"

OUTPUT_FILE = PROCESSED_DIR / "merged_monthly_mo.csv"


def main():
    spark = (
        SparkSession.builder
        .appName("BuildMonthlyDatasetWithSpark")
        .getOrCreate()
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Using files:")
    print("S&P 500:", SP500_FILE)
    print("Unemployment:", UNRATE_FILE)
    print("Tax:", TAX_FILE)
    print("Output:", OUTPUT_FILE)

    sp500 = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(SP500_FILE))
    )

    unrate = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(UNRATE_FILE))
    )

    tax = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(TAX_FILE))
    )

    print("S&P 500 columns:", sp500.columns)
    print("Unemployment columns:", unrate.columns)
    print("Tax columns:", tax.columns)

    # 1. Zet alle datums naar maand-start
    sp500 = (
        sp500
        .withColumn("date", to_date(col("date")))
        .withColumn("month", date_trunc("month", col("date")).cast("date"))
    )

    unrate = (
        unrate
        .withColumn("date", to_date(col("date")))
        .withColumn("month", date_trunc("month", col("date")).cast("date"))
    )

    tax = (
        tax
        .withColumn("date", to_date(col("date")))
        .withColumn("month", date_trunc("month", col("date")).cast("date"))
    )

    # 2. S&P 500 op maandniveau
    sp500_monthly = (
        sp500
        .groupBy("month")
        .agg(
            last("close", ignorenulls=True).alias("close")
        )
        .orderBy("month")
    )

    w_month = Window.orderBy("month")

    sp500_monthly = sp500_monthly.withColumn(
        "previous_close",
        last("close", ignorenulls=True).over(w_month.rowsBetween(-1, -1))
    )

    sp500_monthly = sp500_monthly.withColumn(
        "pct_change",
        ((col("close") - col("previous_close")) / col("previous_close")) * 100
    )

    sp500_monthly = sp500_monthly.drop("previous_close")

    # 3. Unemployment op maandniveau
    unrate_monthly = (
        unrate
        .groupBy("month")
        .agg(
            last("UNRATE", ignorenulls=True).alias("unemployment_rate")
        )
        .orderBy("month")
    )

    # 4. Tax revenue, quarterly naar monthly met forward fill
    tax_quarterly = (
        tax
        .groupBy("month")
        .agg(
            last("W006RC1Q027SBEA", ignorenulls=True).alias("receipts_bn")
        )
        .orderBy("month")
    )

    # Deze reeks is quarterly SAAR. Valerie/Jules hun pipeline deelt door 12
    # om een monthly equivalent te krijgen.
    tax_quarterly = tax_quarterly.withColumn(
        "receipts_bn",
        col("receipts_bn") / 12
    )

    monthly_index = sp500_monthly.select("month").distinct().orderBy("month")

    tax_monthly = (
        monthly_index
        .join(tax_quarterly, on="month", how="left")
        .orderBy("month")
    )

    w_ffill = (
        Window
        .orderBy("month")
        .rowsBetween(Window.unboundedPreceding, 0)
    )

    tax_monthly = tax_monthly.withColumn(
        "receipts_bn",
        last("receipts_bn", ignorenulls=True).over(w_ffill)
    )

    # 5. Join alle datasets
    merged = (
        sp500_monthly
        .join(unrate_monthly, on="month", how="inner")
        .join(tax_monthly, on="month", how="inner")
        .orderBy("month")
    )

    merged = merged.withColumnRenamed("month", "date")

    # 6. Maak change variables
    w_final = Window.orderBy("date")

    merged = merged.withColumn(
        "previous_unemployment_rate",
        last("unemployment_rate", ignorenulls=True).over(w_final.rowsBetween(-1, -1))
    )

    merged = merged.withColumn(
        "previous_receipts_bn",
        last("receipts_bn", ignorenulls=True).over(w_final.rowsBetween(-1, -1))
    )

    merged = merged.withColumn(
        "pp_change_unrate",
        col("unemployment_rate") - col("previous_unemployment_rate")
    )

    merged = merged.withColumn(
        "pct_change_receipts",
        ((col("receipts_bn") - col("previous_receipts_bn")) / col("previous_receipts_bn")) * 100
    )

    merged = merged.drop(
        "previous_unemployment_rate",
        "previous_receipts_bn"
    )

    # 7. Opslaan als gewone CSV
    merged_pd = merged.toPandas()

    merged_pd = merged_pd[
        [
            "date",
            "close",
            "unemployment_rate",
            "receipts_bn",
            "pct_change",
            "pp_change_unrate",
            "pct_change_receipts",
        ]
    ]

    merged_pd = merged_pd.dropna(subset=[
        "close",
        "unemployment_rate",
        "receipts_bn",
    ])

    merged_pd.to_csv(OUTPUT_FILE, index=False)

    print("Saved Spark-built dataset to:", OUTPUT_FILE)
    print("Rows:", len(merged_pd))
    print("Columns:", merged_pd.columns.tolist())
    print(merged_pd.head())
    print(merged_pd.tail())

    spark.stop()


if __name__ == "__main__":
    main()