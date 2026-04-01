"""
Script om visualisaties te maken van de raw en processed data.
Genereert plots in de outputs/figures/ directory.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def set_plot_style():
    """Stel een mooie plot style in."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_raw_data_overview(project_root: Path):
    """Plot overzicht van alle raw data."""
    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lees alle raw data
    sp500 = pd.read_csv(raw_dir / "github_sp500_daily.csv", index_col='date', parse_dates=True)
    unrate = pd.read_csv(raw_dir / "fred_unrate.csv", index_col='date', parse_dates=True)
    tax = pd.read_csv(raw_dir / "fred_w006rc1q027sbea.csv", index_col='date', parse_dates=True)
    
    # Hernoem kolommen voor consistentie
    unrate = unrate.rename(columns={'UNRATE': 'value'})
    tax = tax.rename(columns={'W006RC1Q027SBEA': 'value'})
    
    # Plot 1: Alle drie de datasets samen over tijd
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Raw Data Overzicht - Volledige Tijdsreeksen', fontsize=16, fontweight='bold')
    
    # S&P 500
    axes[0].plot(sp500.index, sp500['close'], color='darkblue', linewidth=1.5)
    axes[0].set_ylabel('S&P 500 Prijs', fontweight='bold')
    axes[0].set_title(f'S&P 500 ({sp500.index.min().year} - {sp500.index.max().year})')
    axes[0].grid(True, alpha=0.3)
    
    # Werkloosheid
    axes[1].plot(unrate.index, unrate['value'], color='darkred', linewidth=1.5)
    axes[1].set_ylabel('Werkloosheid (%)', fontweight='bold')
    axes[1].set_title(f'Werkloosheidspercentage ({unrate.index.min().year} - {unrate.index.max().year})')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=unrate['value'].mean(), color='red', linestyle='--', alpha=0.5, label=f'Gemiddelde: {unrate["value"].mean():.1f}%')
    axes[1].legend()
    
    # Belastingopbrengsten
    axes[2].plot(tax.index, tax['value'], color='darkgreen', linewidth=1.5, marker='o', markersize=2)
    axes[2].set_ylabel('Belastingopbrengsten (miljard $)', fontweight='bold')
    axes[2].set_xlabel('Jaar', fontweight='bold')
    axes[2].set_title(f'Federale Inkomstenbelasting Opbrengsten ({tax.index.min().year} - {tax.index.max().year})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_raw_data_overview.png', dpi=300, bbox_inches='tight')
    print(f"✅ Opgeslagen: 01_raw_data_overview.png")
    plt.close()
    
    # Plot 2: S&P 500 met log scale (om groei beter te zien)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.semilogy(sp500.index, sp500['close'], color='darkblue', linewidth=1.5)
    ax.set_ylabel('S&P 500 Prijs (log scale)', fontweight='bold')
    ax.set_xlabel('Jaar', fontweight='bold')
    ax.set_title('S&P 500 Historische Ontwikkeling (Logaritmische Schaal)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '02_sp500_log_scale.png', dpi=300, bbox_inches='tight')
    print(f"✅ Opgeslagen: 02_sp500_log_scale.png")
    plt.close()


def plot_processed_data_overview(project_root: Path):
    """Plot overzicht van de processed combined data."""
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "figures"
    
    df = pd.read_csv(processed_dir / "combined_full_dataset.csv", index_col='date', parse_dates=True)
    
    # Plot 3: Combined data - alle variabelen
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Gecombineerde Dataset (1949-2025)', fontsize=16, fontweight='bold')
    
    # S&P 500 met downturns gemarkeerd
    axes[0].plot(df.index, df['sp500_adjusted'], color='darkblue', linewidth=1.5, label='S&P 500')
    downturn_dates = df[df['downturn'] == 1].index
    axes[0].scatter(downturn_dates, df.loc[downturn_dates, 'sp500_adjusted'], 
                   color='red', s=50, zorder=5, label='Downturn (≤-5%)', alpha=0.7)
    axes[0].set_ylabel('S&P 500 Prijs', fontweight='bold')
    axes[0].set_title('S&P 500 met Downturns Gemarkeerd')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Werkloosheid
    axes[1].plot(df.index, df['unemployment_rate'], color='darkred', linewidth=1.5)
    axes[1].fill_between(df.index, df['unemployment_rate'], alpha=0.3, color='red')
    axes[1].set_ylabel('Werkloosheid (%)', fontweight='bold')
    axes[1].set_title('Werkloosheidspercentage')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=df['unemployment_rate'].mean(), color='red', linestyle='--', 
                   alpha=0.5, label=f'Gemiddelde: {df["unemployment_rate"].mean():.1f}%')
    axes[1].legend()
    
    # Belastingopbrengsten
    axes[2].plot(df.index, df['federal_income_tax_revenue'], color='darkgreen', linewidth=1.5)
    axes[2].fill_between(df.index, df['federal_income_tax_revenue'], alpha=0.3, color='green')
    axes[2].set_ylabel('Belastingopbrengsten (miljard $)', fontweight='bold')
    axes[2].set_xlabel('Jaar', fontweight='bold')
    axes[2].set_title('Federale Inkomstenbelasting Opbrengsten')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_combined_data_timeseries.png', dpi=300, bbox_inches='tight')
    print(f"✅ Opgeslagen: 03_combined_data_timeseries.png")
    plt.close()
    
    # Plot 4: Maandelijkse returns distributie
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('S&P 500 Maandelijkse Returns Analyse', fontsize=16, fontweight='bold')
    
    # Histogram
    axes[0].hist(df['sp500_monthly_return'].dropna() * 100, bins=50, 
                color='darkblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=-5, color='red', linestyle='--', linewidth=2, label='Downturn threshold (-5%)')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Maandelijks Rendement (%)', fontweight='bold')
    axes[0].set_ylabel('Frequentie', fontweight='bold')
    axes[0].set_title('Distributie van Maandelijkse Returns')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Time series van returns met threshold
    axes[1].plot(df.index, df['sp500_monthly_return'] * 100, color='darkblue', linewidth=0.8, alpha=0.6)
    axes[1].axhline(y=-5, color='red', linestyle='--', linewidth=2, label='Downturn threshold (-5%)')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].scatter(downturn_dates, df.loc[downturn_dates, 'sp500_monthly_return'] * 100,
                   color='red', s=30, zorder=5, alpha=0.7)
    axes[1].set_ylabel('Maandelijks Rendement (%)', fontweight='bold')
    axes[1].set_xlabel('Jaar', fontweight='bold')
    axes[1].set_title('Maandelijkse Returns over Tijd')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_returns_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Opgeslagen: 04_returns_analysis.png")
    plt.close()
    
    # Plot 5: Downturns over decennia
    fig, ax = plt.subplots(figsize=(12, 6))
    downturns = df[df['downturn'] == 1]
    downturns_by_decade = downturns.groupby(downturns.index.year // 10 * 10).size()
    
    decades = downturns_by_decade.index
    counts = downturns_by_decade.values
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(decades)))
    
    bars = ax.bar(decades, counts, width=8, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Decennium', fontweight='bold')
    ax.set_ylabel('Aantal Downturns', fontweight='bold')
    ax.set_title('Aantal Downturns (≤-5%) per Decennium', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Voeg waarden toe op de bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_downturns_by_decade.png', dpi=300, bbox_inches='tight')
    print(f"✅ Opgeslagen: 05_downturns_by_decade.png")
    plt.close()
    
    # Plot 6: Correlatie tussen variabelen
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Scatter Plots - Relaties tussen Variabelen', fontsize=16, fontweight='bold')
    
    # S&P 500 vs Werkloosheid
    axes[0, 0].scatter(df['sp500_adjusted'], df['unemployment_rate'], 
                      alpha=0.5, s=20, color='purple')
    axes[0, 0].set_xlabel('S&P 500 Prijs', fontweight='bold')
    axes[0, 0].set_ylabel('Werkloosheid (%)', fontweight='bold')
    axes[0, 0].set_title('S&P 500 vs Werkloosheid')
    axes[0, 0].grid(True, alpha=0.3)
    
    # S&P 500 returns vs Werkloosheid change
    df_temp = df.copy()
    df_temp['unemployment_change'] = df_temp['unemployment_rate'].diff()
    axes[0, 1].scatter(df_temp['sp500_monthly_return'] * 100, df_temp['unemployment_change'],
                      alpha=0.5, s=20, color='orange')
    axes[0, 1].axvline(x=-5, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('S&P 500 Maandelijks Rendement (%)', fontweight='bold')
    axes[0, 1].set_ylabel('Verandering Werkloosheid (%-punt)', fontweight='bold')
    axes[0, 1].set_title('Returns vs Werkloosheidsverandering')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Werkloosheid vs Belastingopbrengsten
    axes[1, 0].scatter(df['unemployment_rate'], df['federal_income_tax_revenue'],
                      alpha=0.5, s=20, color='green')
    axes[1, 0].set_xlabel('Werkloosheid (%)', fontweight='bold')
    axes[1, 0].set_ylabel('Belastingopbrengsten (miljard $)', fontweight='bold')
    axes[1, 0].set_title('Werkloosheid vs Belastingopbrengsten')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time trend: alle drie genormaliseerd
    df_norm = df[['sp500_adjusted', 'unemployment_rate', 'federal_income_tax_revenue']].copy()
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
    
    axes[1, 1].plot(df_norm.index, df_norm['sp500_adjusted'], label='S&P 500 (norm)', alpha=0.7, linewidth=1.5)
    axes[1, 1].plot(df_norm.index, df_norm['unemployment_rate'], label='Werkloosheid (norm)', alpha=0.7, linewidth=1.5)
    axes[1, 1].plot(df_norm.index, df_norm['federal_income_tax_revenue'], label='Belastingen (norm)', alpha=0.7, linewidth=1.5)
    axes[1, 1].set_ylabel('Gestandaardiseerde Waarde', fontweight='bold')
    axes[1, 1].set_xlabel('Jaar', fontweight='bold')
    axes[1, 1].set_title('Genormaliseerde Tijdreeksen')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_variable_relationships.png', dpi=300, bbox_inches='tight')
    print(f"✅ Opgeslagen: 06_variable_relationships.png")
    plt.close()


def create_summary_statistics(project_root: Path):
    """Maak een tabel met samenvatting statistieken."""
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(processed_dir / "combined_full_dataset.csv", index_col='date', parse_dates=True)
    
    # Basis statistieken
    stats = df[['sp500_adjusted', 'sp500_monthly_return', 'unemployment_rate', 'federal_income_tax_revenue']].describe()
    stats.loc['median'] = df[['sp500_adjusted', 'sp500_monthly_return', 'unemployment_rate', 'federal_income_tax_revenue']].median()
    
    # Hernoem kolommen voor duidelijkheid
    stats = stats.rename(columns={
        'sp500_adjusted': 'S&P 500',
        'sp500_monthly_return': 'Maandelijks Rendement',
        'unemployment_rate': 'Werkloosheid (%)',
        'federal_income_tax_revenue': 'Belastingopbrengsten (mld $)'
    })
    
    stats.to_csv(output_dir / 'summary_statistics.csv')
    print(f"✅ Opgeslagen: summary_statistics.csv")
    
    # Downturn statistieken
    downturn_stats = pd.DataFrame({
        'Totaal aantal maanden': [len(df)],
        'Aantal downturns': [(df['downturn'] == 1).sum()],
        'Percentage downturns': [(df['downturn'] == 1).sum() / len(df) * 100],
        'Gemiddelde return tijdens downturn': [df[df['downturn'] == 1]['sp500_monthly_return'].mean() * 100],
        'Gemiddelde return (alle maanden)': [df['sp500_monthly_return'].mean() * 100],
    }).T
    downturn_stats.columns = ['Waarde']
    downturn_stats.to_csv(output_dir / 'downturn_statistics.csv')
    print(f"✅ Opgeslagen: downturn_statistics.csv")
    
    return stats, downturn_stats


def main():
    """Hoofdfunctie om alle visualisaties te maken."""
    project_root = Path(__file__).resolve().parents[1]
    
    print("=" * 60)
    print("DATA VISUALISATIE SCRIPT")
    print("=" * 60)
    print()
    
    set_plot_style()
    
    print("📊 Genereren van plots voor RAW data...")
    plot_raw_data_overview(project_root)
    print()
    
    print("📊 Genereren van plots voor PROCESSED data...")
    plot_processed_data_overview(project_root)
    print()
    
    print("📋 Genereren van statistiek tabellen...")
    stats, downturn_stats = create_summary_statistics(project_root)
    print()
    
    print("=" * 60)
    print("✅ ALLE VISUALISATIES ZIJN AANGEMAAKT!")
    print("=" * 60)
    print()
    print("📁 Locaties:")
    print(f"   - Figuren: {project_root}/outputs/figures/")
    print(f"   - Tabellen: {project_root}/outputs/tables/")
    print()
    print("📈 Aangemaakte plots:")
    print("   1. 01_raw_data_overview.png - Overzicht alle raw data")
    print("   2. 02_sp500_log_scale.png - S&P 500 logaritmische schaal")
    print("   3. 03_combined_data_timeseries.png - Combined data tijdreeksen")
    print("   4. 04_returns_analysis.png - Returns distributie en analyse")
    print("   5. 05_downturns_by_decade.png - Downturns per decennium")
    print("   6. 06_variable_relationships.png - Relaties tussen variabelen")
    print()
    print("📊 Statistieken:")
    print("\nBasisstatistieken:")
    print(stats.round(2))
    print("\nDownturn statistieken:")
    print(downturn_stats.round(2))


if __name__ == "__main__":
    main()
