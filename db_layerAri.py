"""
High-level database layer for persisting raw market data into SQLite.

"""
from pathlib import Path
from typing import Optional

import pandas as pd

from db_utils import (
    get_db_path,
    get_connection,
    table_exists,
    get_table_row_count,
    list_tables,
)

RAW_SP500_TABLE = "sp500_daily"
RAW_UNRATE_TABLE = "unemployment"
RAW_TAX_TABLE = "tax_revenue"


def init_database(project_root: Optional[Path] = None) -> Path:
    """Ensure the SQLite database file exists and return its path."""
    db_path = get_db_path(project_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    conn.close()
    return db_path


def store_dataframe(
    df: pd.DataFrame,
    table_name: str,
    project_root: Optional[Path] = None,
    if_exists: str = "replace",
) -> None:
    """Store a DataFrame into the SQLite database."""
    db_path = get_db_path(project_root)
    conn = get_connection(db_path)
    try:
        df_to_store = df.copy()
        # Save index as 'date' column if it’s a time index
        if isinstance(df_to_store.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            idx_name = df_to_store.index.name or "index"
            df_to_store = df_to_store.reset_index().rename(columns={idx_name: "date"})
        df_to_store.to_sql(table_name, conn, if_exists=if_exists, index=False)
    finally:
        conn.close()


def store_raw_sp500(sp500_daily: pd.DataFrame, project_root: Optional[Path] = None) -> None:
    """Persist raw S&P 500 daily data into table sp500_daily."""
    store_dataframe(sp500_daily, RAW_SP500_TABLE, project_root=project_root)


def store_raw_unemployment(
    unrate: pd.Series | pd.DataFrame,
    project_root: Optional[Path] = None,
) -> None:
    """Persist raw unemployment series into table unemployment."""
    if isinstance(unrate, pd.Series):
        df = unrate.to_frame(name=unrate.name or "value")
    else:
        df = unrate
    store_dataframe(df, RAW_UNRATE_TABLE, project_root=project_root)


def store_raw_tax_revenue(
    tax: pd.Series | pd.DataFrame,
    project_root: Optional[Path] = None,
) -> None:
    """Persist raw tax revenue series into table tax_revenue."""
    if isinstance(tax, pd.Series):
        df = tax.to_frame(name=tax.name or "value")
    else:
        df = tax
    store_dataframe(df, RAW_TAX_TABLE, project_root=project_root)


def debug_print_overview(project_root: Optional[Path] = None) -> None:
    """Print a short overview of tables and row counts (for demo/report)."""
    db_path = get_db_path(project_root)
    print(f"SQLite database: {db_path}")
    tables = list_tables(db_path)
    if not tables:
        print("No tables found yet.")
        return
    for name in tables:
        count = get_table_row_count(name, db_path)
        print(f"- {name}: {count} rows")
