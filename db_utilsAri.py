"""
Database utility functions for SQLite operations.
"""
from pathlib import Path
import sqlite3
from typing import Optional
import pandas as pd


def get_db_path(project_root: Optional[Path] = None) -> Path:
    """Get the path to the SQLite database.

    The database is stored in data/raw/market_data.db under the project root.
    """
    if project_root is None:
        # assume we're somewhere inside src/, e.g. src/ or src/00_ingest
        project_root = Path(__file__).resolve().parents[1]

    db_path = project_root / "data" / "raw" / "market_data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite database connection.

    If db_path is not provided, use the default path under data/raw.
    """
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def query_to_dataframe(
    query: str,
    db_path: Optional[Path] = None,
    parse_dates: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(query, conn, parse_dates=parse_dates)
    finally:
        conn.close()
    return df


def table_exists(table_name: str, db_path: Optional[Path] = None) -> bool:
    """Check if a table exists in the database."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def get_table_row_count(table_name: str, db_path: Optional[Path] = None) -> int:
    """Get the number of rows in a table."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        (count,) = cur.fetchone()
        return int(count)
    finally:
        conn.close()


def list_tables(db_path: Optional[Path] = None) -> list[str]:
    """List all tables in the database, ordered by name."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
