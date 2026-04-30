"""
Database utility functions for SQLite operations.
"""
from pathlib import Path
import sqlite3
from typing import Optional
import pandas as pd


def get_db_path(project_root: Optional[Path] = None) -> Path:
    """Get the path to the SQLite database."""
    if project_root is None:
        # Assume we're in src/00_ingest or similar
        project_root = Path(__file__).resolve().parents[2]
    
    db_path = project_root / "data" / "raw" / "market_data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite database connection."""
    if db_path is None:
        db_path = get_db_path()
    return sqlite3.connect(db_path)


def query_to_dataframe(
    query: str,
    db_path: Optional[Path] = None,
    parse_dates: Optional[list] = None,
) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.
    
    Args:
        query: SQL query string
        db_path: Path to SQLite database (optional)
        parse_dates: List of column names to parse as dates (optional)
    
    Returns:
        DataFrame with query results
    """
    conn = get_connection(db_path)
    df = pd.read_sql_query(query, conn, parse_dates=parse_dates)
    conn.close()
    return df


def table_exists(table_name: str, db_path: Optional[Path] = None) -> bool:
    """Check if a table exists in the database."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def get_table_row_count(table_name: str, db_path: Optional[Path] = None) -> int:
    """Get the number of rows in a table."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    conn.close()
    return count


def list_tables(db_path: Optional[Path] = None) -> list[str]:
    """List all tables in the database."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()
    return tables
