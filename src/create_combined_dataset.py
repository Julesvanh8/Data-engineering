"""
Script om een volledig gecombineerd bestand te maken van alle drie de databronnen.
Gebruikt de gemeenschappelijke periode waar alle datasets data hebben.
"""

from pathlib import Path
import pandas as pd


def create_combined_dataset(project_root: Path) -> Path:
    """
    Maak een gecombineerd bestand met alle drie de databronnen.
    """
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Lees de drie datasets
    print("Data inlezen...")
    sp500 = pd.read_csv(raw_dir / "github_sp500_daily.csv", index_col='date', parse_dates=True)
    unrate = pd.read_csv(raw_dir / "fred_unrate.csv", index_col='date', parse_dates=True)
    tax = pd.read_csv(raw_dir / "fred_w006rc1q027sbea.csv", index_col='date', parse_dates=True)
    
    # Hernoem kolommen voor duidelijkheid
    sp500 = sp500.rename(columns={'close': 'sp500_price', 'adjusted_close': 'sp500_adjusted'})
    sp500 = sp500[['sp500_price', 'sp500_adjusted']]  # Drop volume kolom
    
    unrate = unrate.rename(columns={'value': 'unemployment_rate'})
    tax = tax.rename(columns={'value': 'federal_income_tax_revenue'})
    
    # Vind de gemeenschappelijke periode
    start_date = max(sp500.index.min(), unrate.index.min(), tax.index.min())
    end_date = min(sp500.index.max(), unrate.index.max(), tax.index.max())
    
    print(f"\nGemeenschappelijke periode: {start_date.date()} tot {end_date.date()}")
    
    # Filter datasets op gemeenschappelijke periode
    sp500_filtered = sp500[(sp500.index >= start_date) & (sp500.index <= end_date)]
    unrate_filtered = unrate[(unrate.index >= start_date) & (unrate.index <= end_date)]
    tax_filtered = tax[(tax.index >= start_date) & (tax.index <= end_date)]
    
    print(f"  S&P 500: {len(sp500_filtered)} observaties")
    print(f"  Werkloosheid: {len(unrate_filtered)} observaties")
    print(f"  Belastingen: {len(tax_filtered)} observaties (kwartaaldata)")
    
    # Resample belastingdata (kwartaal) naar maandelijks met forward fill
    tax_monthly = tax_filtered.resample('MS').ffill()
    print(f"  Belastingen na resample: {len(tax_monthly)} observaties")
    
    # Start met S&P 500 data
    combined = sp500_filtered.copy()
    
    # Join met werkloosheid
    combined = combined.join(unrate_filtered, how='inner')
    print(f"\nNa join met werkloosheid: {len(combined)} observaties")
    
    # Join met belastingen
    combined = combined.join(tax_monthly, how='inner')
    print(f"Na join met belastingen: {len(combined)} observaties")
    
    # Hernoem de FRED kolommen naar meer duidelijke namen
    combined = combined.rename(columns={
        'UNRATE': 'unemployment_rate',
        'W006RC1Q027SBEA': 'federal_income_tax_revenue'
    })
    
    # Bereken maandelijkse return
    combined['sp500_monthly_return'] = combined['sp500_adjusted'].pct_change()
    
    # Definieer downturn (standaard: -5% of meer)
    downturn_threshold = -0.05
    combined['downturn'] = (combined['sp500_monthly_return'] <= downturn_threshold).astype(int)
    
    # Voeg lag features toe voor downturn (1-12 maanden)
    for lag in range(1, 13):
        combined[f'downturn_lag_{lag}m'] = combined['downturn'].shift(lag)
    
    # Verwijder rijen met NaN waarden
    combined_clean = combined.dropna()
    
    print(f"\nGecombineerde dataset:")
    print(f"  Aantal observaties: {len(combined_clean)}")
    print(f"  Aantal kolommen: {len(combined_clean.columns)}")
    print(f"  Periode: {combined_clean.index.min().date()} tot {combined_clean.index.max().date()}")
    print(f"\nKolommen: {list(combined_clean.columns)}")
    
    # Sla op
    output_path = processed_dir / "combined_full_dataset.csv"
    combined_clean.to_csv(output_path)
    print(f"\n✅ Dataset opgeslagen: {output_path}")
    
    # Toon een preview
    print("\nVoorbeeld data (eerste 5 rijen):")
    print(combined_clean.head())
    
    # Toon statistieken
    print("\nBasisstatistieken:")
    print(combined_clean.describe())
    
    return output_path


def main():
    project_root = Path(__file__).resolve().parents[1]
    create_combined_dataset(project_root)


if __name__ == "__main__":
    main()
