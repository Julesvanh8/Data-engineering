"""
Run all data ingestion scripts to populate the SQLite database.
This script coordinates fetching data from all sources and storing in SQLite.
"""
from pathlib import Path
import sys

# Add src directory to path for imports
src_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(src_dir))

from ingest_sp500 import main as ingest_sp500
from ingest_unemployment import main as ingest_unemployment
from ingest_tax_revenue import main as ingest_tax_revenue
from db_utils import list_tables, get_table_row_count, get_db_path


def main():
    """Run all ingestion scripts in sequence."""
    print("=" * 70)
    print("INGESTION PIPELINE - Fetching all data sources")
    print("=" * 70)
    
    # 1. Ingest S&P 500 data
    print("\n[1/3] S&P 500 Data")
    print("-" * 70)
    try:
        ingest_sp500()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # 2. Ingest unemployment data
    print("\n[2/3] Unemployment Data")
    print("-" * 70)
    try:
        ingest_unemployment()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # 3. Ingest tax revenue data
    print("\n[3/3] Tax Revenue Data")
    print("-" * 70)
    try:
        ingest_tax_revenue()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    
    db_path = get_db_path()
    print(f"\nDatabase: {db_path}")
    print("\nTables created:")
    
    for table in list_tables():
        count = get_table_row_count(table)
        print(f"  - {table}: {count:,} rows")
    
    print("\n✅ All data successfully ingested!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
