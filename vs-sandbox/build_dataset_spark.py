"""
spark_build_dataset.py — PySpark version of the monthly dataset builder.

Uses PySpark for the ETL build step:
  1. Loads S&P 500, unemployment, and federal tax revenue from SQLite.
  2. Harmonises all series to a monthly date index.
  3. Forward-fills quarterly tax revenue to monthly frequency.
  4. Joins the sources into one dataset.
  5. Creates the change variables used in the lag analysis:
       - pct_change
       - pp_change_unrate
       - pct_change_receipts
  6. Saves the final dataset as:
       data/processed/merged_monthly.csv
"""

from pathlib import Path
import sqlite3
import pandas as pd
import sys
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, date_trunc, last
from pyspark.sql.window import Window


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DB_PATH = RAW_DIR / "market_data.db"
OUTPUT_FILE = PROCESSED_DIR / "merged_monthly.csv"


def load_table_from_sqlite(query: str) -> pd.DataFrame:
    """
    Load a SQLite query result into a pandas DataFrame.
    Spark does not read SQLite directly without a JDBC driver,
    so we read from SQLite with pandas and then convert to Spark.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run the ingestion pipeline first."
        )

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def main():
    spark = (
        SparkSession.builder
        .appName("BuildMonthlyDatasetWithSpark")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Using SQLite database:")
    print("Database:", DB_PATH)
    print("Output:", OUTPUT_FILE)

    # 1. Load data from SQLite
    sp500_pd = load_table_from_sqlite("""
        SELECT date, close
        FROM sp500
        ORDER BY date
    """)

    unrate_pd = load_table_from_sqlite("""
        SELECT date, rate AS unemployment_rate
        FROM unemployment
        ORDER BY date
    """)

    tax_pd = load_table_from_sqlite("""
        SELECT date, receipts_bn
        FROM tax_revenue
        WHERE source = 'FRED_quarterly'
        ORDER BY date
    """)

    print("Rows loaded from SQLite:")
    print("S&P 500:", len(sp500_pd))
    print("Unemployment:", len(unrate_pd))
    print("Tax:", len(tax_pd))

    # 2. Convert pandas DataFrames to Spark DataFrames
    sp500 = spark.createDataFrame(sp500_pd)
    unrate = spark.createDataFrame(unrate_pd)
    tax = spark.createDataFrame(tax_pd)

    print("S&P 500 columns:", sp500.columns)
    print("Unemployment columns:", unrate.columns)
    print("Tax columns:", tax.columns)

    # 3. Convert all dates to month-start
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

    # 4. S&P 500 at monthly level
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

    # 5. Unemployment at monthly level
    unrate_monthly = (
        unrate
        .groupBy("month")
        .agg(
            last("unemployment_rate", ignorenulls=True).alias("unemployment_rate")
        )
        .orderBy("month")
    )

    # 6. Tax revenue, quarterly to monthly with forward fill
    tax_quarterly = (
        tax
        .groupBy("month")
        .agg(
            last("receipts_bn", ignorenulls=True).alias("receipts_bn")
        )
        .orderBy("month")
    )

    # FRED tax revenue is quarterly. We divide by 12 to create a monthly equivalent.
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

    # 7. Join all datasets
    merged = (
        sp500_monthly
        .join(unrate_monthly, on="month", how="inner")
        .join(tax_monthly, on="month", how="inner")
        .orderBy("month")
    )

    merged = merged.withColumnRenamed("month", "date")

    # 8. Create change variables
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

    # 9. Save as normal CSV
    merged_pd = merged.toPandas()

    merged_pd = merged_pd[[
        "date",
        "close",
        "unemployment_rate",
        "receipts_bn",
        "pct_change",
        "pp_change_unrate",
        "pct_change_receipts",
    ]]

    merged_pd = merged_pd.dropna(subset=[
        "close",
        "unemployment_rate",
        "receipts_bn",
    ])

    merged_pd["date"] = pd.to_datetime(merged_pd["date"])
    merged_pd = merged_pd.sort_values("date")

    if "date" not in merged_pd.columns:
        raise ValueError(f"No date column before saving. Columns are: {merged_pd.columns.tolist()}")

    print("Final output columns before saving:")
    print(merged_pd.columns.tolist())

    merged_pd.to_csv(OUTPUT_FILE, index=False)

    print("Saved Spark-built dataset to:", OUTPUT_FILE)
    print("Rows:", len(merged_pd))
    print("Columns:", merged_pd.columns.tolist())
    print(merged_pd.head())
    print(merged_pd.tail())

    spark.stop()


if __name__ == "__main__":
    main()