"""
build_dataset_spark.py — PySpark version of build_dataset.py.

Uses PySpark for the ETL pipeline:
  1. Loads S&P 500, unemployment, and tax revenue from CSV
  2. Forward-fills quarterly tax data to monthly using a window function
  3. Joins everything to a common monthly index
  4. Derives computed columns (pct_change, pp_change_unrate, pct_change_receipts)
     using Spark window functions
  5. Outputs data/processed/merged_monthly_vs.parquet

Usage:
    python vs-sandbox/build_dataset_spark.py
    (or called via run_all.py when user selects Spark)
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pathlib import Path

_HADOOP_HOME = r"C:\Users\schil100\Documents\hadoop"
_HADOOP_BIN  = _HADOOP_HOME + r"\bin"
os.environ["HADOOP_HOME"] = _HADOOP_HOME
os.environ["PATH"] = _HADOOP_BIN + os.pathsep + os.environ.get("PATH", "")

_PYTHON = str(Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe")
os.environ["PYSPARK_PYTHON"]        = _PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = _PYTHON


SHILLER_CSV     = Path(__file__).resolve().parents[1] / "data" / "raw" / "github_sp500_daily.csv"
FRED_UNRATE_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "fred_unrate.csv"
FRED_TAX_CSV    = Path(__file__).resolve().parents[1] / "data" / "raw" / "fred_w006rc1q027sbea.csv"
PROCESSED_DIR   = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_PATH     = PROCESSED_DIR / "merged_monthly_vs.parquet"


def build_dataset() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("build_dataset")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("Loading sources...")

        sp500 = (
            spark.read.csv(str(SHILLER_CSV), header=True, inferSchema=True)
            .withColumn("date", F.trunc(F.to_date("date"), "month"))
            .select(F.col("date"), F.col("close").cast("double"))
            .orderBy("date")
        )

        unemp = (
            spark.read.csv(str(FRED_UNRATE_CSV), header=True, inferSchema=True)
            .withColumn("date", F.trunc(F.to_date("date"), "month"))
            .withColumnRenamed("UNRATE", "unemployment_rate")
            .select("date", F.col("unemployment_rate").cast("double"))
        )

        tax_raw = (
            spark.read.csv(str(FRED_TAX_CSV), header=True, inferSchema=True)
            .withColumn("date", F.trunc(F.to_date("date"), "month"))
            .withColumn("receipts_bn", F.col("W006RC1Q027SBEA").cast("double") / 12)
            .select("date", "receipts_bn")
        )

        # ── forward-fill quarterly tax to monthly frequency ──────────────────
        bounds = tax_raw.agg(
            F.min("date").alias("min_d"),
            F.max("date").alias("max_d"),
        ).collect()[0]

        monthly_spine = spark.sql(f"""
            SELECT explode(sequence(
                date '{bounds["min_d"]}',
                date '{bounds["max_d"]}',
                interval 1 month
            )) AS date
        """)

        w_ffill = Window.orderBy("date").rowsBetween(Window.unboundedPreceding, 0)
        tax = (
            monthly_spine
            .join(tax_raw, on="date", how="left")
            .withColumn("receipts_bn", F.last("receipts_bn", ignorenulls=True).over(w_ffill))
            .filter(F.col("receipts_bn").isNotNull())
        )

        print(f"  S&P 500:       {sp500.count()} rows")
        print(f"  Unemployment:  {unemp.count()} rows")
        print(f"  Tax (monthly): {tax.count()} rows")

        print("\nJoining sources...")
        df = (
            sp500
            .join(unemp, on="date", how="left")
            .join(tax,   on="date", how="left")
            .filter(
                F.col("unemployment_rate").isNotNull() &
                F.col("receipts_bn").isNotNull()
            )
            .orderBy("date")
        )

        w = Window.orderBy("date")
        df = (
            df
            .withColumn("pct_change",
                (F.col("close") - F.lag("close", 1).over(w)) /
                F.lag("close", 1).over(w) * 100)
            .withColumn("pp_change_unrate",
                F.col("unemployment_rate") - F.lag("unemployment_rate", 1).over(w))
            .withColumn("pct_change_receipts",
                (F.col("receipts_bn") - F.lag("receipts_bn", 1).over(w)) /
                F.lag("receipts_bn", 1).over(w) * 100)
        )

        n          = df.count()
        date_range = df.agg(F.min("date"), F.max("date")).collect()[0]
        print(f"  Joined: {n} rows  ({date_range[0]} → {date_range[1]})")

        df.write.mode("overwrite").parquet(str(OUTPUT_PATH))
        print(f"\nSaved to {OUTPUT_PATH}")

    finally:
        spark.stop()


if __name__ == "__main__":
    build_dataset()
