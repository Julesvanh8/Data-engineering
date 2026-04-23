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
    
    # Plot 1: All three datasets over time
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Raw Data Overview - Complete Time Series', fontsize=16, fontweight='bold')
    
    # S&P 500
    axes[0].plot(sp500.index, sp500['close'], color='darkblue', linewidth=1.5)
    axes[0].set_ylabel('S&P 500 Price', fontweight='bold')
    axes[0].set_title(f'S&P 500 ({sp500.index.min().year} - {sp500.index.max().year})')
    axes[0].grid(True, alpha=0.3)
    
    # Unemployment
    axes[1].plot(unrate.index, unrate['value'], color='darkred', linewidth=1.5)
    axes[1].set_ylabel('Unemployment (%)', fontweight='bold')
    axes[1].set_title(f'Unemployment Rate ({unrate.index.min().year} - {unrate.index.max().year})')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=unrate['value'].mean(), color='red', linestyle='--', alpha=0.5, label=f'Average: {unrate["value"].mean():.1f}%')
    axes[1].legend()
    
    # Tax Revenue
    axes[2].plot(tax.index, tax['value'], color='darkgreen', linewidth=1.5, marker='o', markersize=2)
    axes[2].set_ylabel('Tax Revenue (billion $)', fontweight='bold')
    axes[2].set_xlabel('Year', fontweight='bold')
    axes[2].set_title(f'Federal Income Tax Revenue ({tax.index.min().year} - {tax.index.max().year})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_raw_data_overview.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 01_raw_data_overview.png")
    plt.close()
    
    # Plot 2: S&P 500 with log scale (to better show growth)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.semilogy(sp500.index, sp500['close'], color='darkblue', linewidth=1.5)
    ax.set_ylabel('S&P 500 Price (log scale)', fontweight='bold')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_title('S&P 500 Historical Development (Logarithmic Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '02_sp500_log_scale.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 02_sp500_log_scale.png")
    plt.close()


def plot_processed_data_overview(project_root: Path):
    """Plot overview of the processed combined data."""
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "figures"
    
    df = pd.read_csv(processed_dir / "combined_full_dataset.csv", index_col='date', parse_dates=True)
    
    # Plot 3: Combined data - all variables
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Combined Dataset (1949-2025)', fontsize=16, fontweight='bold')
    
    # S&P 500 with downturns marked
    axes[0].plot(df.index, df['sp500_adjusted'], color='darkblue', linewidth=1.5, label='S&P 500')
    downturn_dates = df[df['downturn'] == 1].index
    axes[0].scatter(downturn_dates, df.loc[downturn_dates, 'sp500_adjusted'], 
                   color='red', s=50, zorder=5, label='Downturn (≤-5%)', alpha=0.7)
    axes[0].set_ylabel('S&P 500 Price', fontweight='bold')
    axes[0].set_title('S&P 500 with Downturns Marked')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Unemployment
    axes[1].plot(df.index, df['unemployment_rate'], color='darkred', linewidth=1.5)
    axes[1].fill_between(df.index, df['unemployment_rate'], alpha=0.3, color='red')
    axes[1].set_ylabel('Unemployment (%)', fontweight='bold')
    axes[1].set_title('Unemployment Rate')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=df['unemployment_rate'].mean(), color='red', linestyle='--', 
                   alpha=0.5, label=f'Average: {df["unemployment_rate"].mean():.1f}%')
    axes[1].legend()
    
    # Tax Revenue
    axes[2].plot(df.index, df['federal_income_tax_revenue'], color='darkgreen', linewidth=1.5)
    axes[2].fill_between(df.index, df['federal_income_tax_revenue'], alpha=0.3, color='green')
    axes[2].set_ylabel('Tax Revenue (billion $)', fontweight='bold')
    axes[2].set_xlabel('Year', fontweight='bold')
    axes[2].set_title('Federal Income Tax Revenue')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_combined_data_timeseries.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 03_combined_data_timeseries.png")
    plt.close()
    
    # Plot 4: Monthly returns distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('S&P 500 Monthly Returns Analysis', fontsize=16, fontweight='bold')
    
    # Histogram
    axes[0].hist(df['sp500_monthly_return'].dropna() * 100, bins=50, 
                color='darkblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=-5, color='red', linestyle='--', linewidth=2, label='Downturn threshold (-5%)')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Monthly Return (%)', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Distribution of Monthly Returns')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Time series of returns with threshold
    axes[1].plot(df.index, df['sp500_monthly_return'] * 100, color='darkblue', linewidth=0.8, alpha=0.6)
    axes[1].axhline(y=-5, color='red', linestyle='--', linewidth=2, label='Downturn threshold (-5%)')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].scatter(downturn_dates, df.loc[downturn_dates, 'sp500_monthly_return'] * 100,
                   color='red', s=30, zorder=5, alpha=0.7)
    axes[1].set_ylabel('Monthly Return (%)', fontweight='bold')
    axes[1].set_xlabel('Year', fontweight='bold')
    axes[1].set_title('Monthly Returns Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_returns_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 04_returns_analysis.png")
    plt.close()
    
    # Plot 5: Downturns over decades
    fig, ax = plt.subplots(figsize=(12, 6))
    downturns = df[df['downturn'] == 1]
    downturns_by_decade = downturns.groupby(downturns.index.year // 10 * 10).size()
    
    decades = downturns_by_decade.index
    counts = downturns_by_decade.values
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(decades)))
    
    bars = ax.bar(decades, counts, width=8, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Decade', fontweight='bold')
    ax.set_ylabel('Number of Downturns', fontweight='bold')
    ax.set_title('Number of Downturns (≤-5%) per Decade', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_downturns_by_decade.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 05_downturns_by_decade.png")
    plt.close()
    
    # Plot 6: Correlation between variables
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Scatter Plots - Relationships Between Variables', fontsize=16, fontweight='bold')
    
    # S&P 500 vs Unemployment
    axes[0, 0].scatter(df['sp500_adjusted'], df['unemployment_rate'], 
                      alpha=0.5, s=20, color='purple')
    axes[0, 0].set_xlabel('S&P 500 Price', fontweight='bold')
    axes[0, 0].set_ylabel('Unemployment (%)', fontweight='bold')
    axes[0, 0].set_title('S&P 500 vs Unemployment')
    axes[0, 0].grid(True, alpha=0.3)
    
    # S&P 500 returns vs Unemployment change
    df_temp = df.copy()
    df_temp['unemployment_change'] = df_temp['unemployment_rate'].diff()
    axes[0, 1].scatter(df_temp['sp500_monthly_return'] * 100, df_temp['unemployment_change'],
                      alpha=0.5, s=20, color='orange')
    axes[0, 1].axvline(x=-5, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('S&P 500 Monthly Return (%)', fontweight='bold')
    axes[0, 1].set_ylabel('Change in Unemployment (%-point)', fontweight='bold')
    axes[0, 1].set_title('Returns vs Unemployment Change')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Unemployment vs Tax Revenue
    axes[1, 0].scatter(df['unemployment_rate'], df['federal_income_tax_revenue'],
                      alpha=0.5, s=20, color='green')
    axes[1, 0].set_xlabel('Unemployment (%)', fontweight='bold')
    axes[1, 0].set_ylabel('Tax Revenue (billion $)', fontweight='bold')
    axes[1, 0].set_title('Unemployment vs Tax Revenue')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time trend: all three normalized
    df_norm = df[['sp500_adjusted', 'unemployment_rate', 'federal_income_tax_revenue']].copy()
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
    
    axes[1, 1].plot(df_norm.index, df_norm['sp500_adjusted'], label='S&P 500 (norm)', alpha=0.7, linewidth=1.5)
    axes[1, 1].plot(df_norm.index, df_norm['unemployment_rate'], label='Unemployment (norm)', alpha=0.7, linewidth=1.5)
    axes[1, 1].plot(df_norm.index, df_norm['federal_income_tax_revenue'], label='Tax Revenue (norm)', alpha=0.7, linewidth=1.5)
    axes[1, 1].set_ylabel('Standardized Value', fontweight='bold')
    axes[1, 1].set_xlabel('Year', fontweight='bold')
    axes[1, 1].set_title('Normalized Time Series')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_variable_relationships.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 06_variable_relationships.png")
    plt.close()


def create_summary_statistics(project_root: Path):
    """Create a table with summary statistics."""
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(processed_dir / "combined_full_dataset.csv", index_col='date', parse_dates=True)
    
    # Basic statistics
    stats = df[['sp500_adjusted', 'sp500_monthly_return', 'unemployment_rate', 'federal_income_tax_revenue']].describe()
    stats.loc['median'] = df[['sp500_adjusted', 'sp500_monthly_return', 'unemployment_rate', 'federal_income_tax_revenue']].median()
    
    # Rename columns for clarity
    stats = stats.rename(columns={
        'sp500_adjusted': 'S&P 500',
        'sp500_monthly_return': 'Monthly Return',
        'unemployment_rate': 'Unemployment (%)',
        'federal_income_tax_revenue': 'Tax Revenue (billion $)'
    })
    
    stats.to_csv(output_dir / 'summary_statistics.csv')
    print(f"✅ Saved: summary_statistics.csv")
    
    # Downturn statistics
    downturn_stats = pd.DataFrame({
        'Total number of months': [len(df)],
        'Number of downturns': [(df['downturn'] == 1).sum()],
        'Percentage downturns': [(df['downturn'] == 1).sum() / len(df) * 100],
        'Average return during downturn': [df[df['downturn'] == 1]['sp500_monthly_return'].mean() * 100],
        'Average return (all months)': [df['sp500_monthly_return'].mean() * 100],
    }).T
    downturn_stats.columns = ['Value']
    downturn_stats.to_csv(output_dir / 'downturn_statistics.csv')
    print(f"✅ Saved: downturn_statistics.csv")
    
    return stats, downturn_stats


def main():
    """Main function to create all visualizations."""
    project_root = Path(__file__).resolve().parents[1]
    
    print("=" * 60)
    print("DATA VISUALIZATION SCRIPT")
    print("=" * 60)
    print()
    
    set_plot_style()
    
    print("📊 Generating plots for RAW data...")
    plot_raw_data_overview(project_root)
    print()
    
    print("📊 Generating plots for PROCESSED data...")
    plot_processed_data_overview(project_root)
    print()
    
    print("📋 Generating statistics tables...")
    stats, downturn_stats = create_summary_statistics(project_root)
    print()
    
    print("=" * 60)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("=" * 60)
    print()
    print("📁 Locations:")
    print(f"   - Figures: {project_root}/outputs/figures/")
    print(f"   - Tables: {project_root}/outputs/tables/")
    print()
    print("📈 Created plots:")
    print("   1. 01_raw_data_overview.png - Overview of all raw data")
    print("   2. 02_sp500_log_scale.png - S&P 500 logarithmic scale")
    print("   3. 03_combined_data_timeseries.png - Combined data time series")
    print("   4. 04_returns_analysis.png - Returns distribution and analysis")
    print("   5. 05_downturns_by_decade.png - Downturns per decade")
    print("   6. 06_variable_relationships.png - Relationships between variables")
    print()
    print("📊 Statistics:")
    print("\nBasic statistics:")
    print(stats.round(2))
    print("\nDownturn statistics:")
    print(downturn_stats.round(2))


if __name__ == "__main__":
    main()
