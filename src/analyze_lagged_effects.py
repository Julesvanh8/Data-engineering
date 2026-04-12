"""
Analyse Script: Lagged Effects van Stock Market Downturns
Onderzoeksvraag: How long after a U.S. stock market downturn do unemployment 
                 and federal income tax revenues change?

Dit script analyseert:
1. Veranderingen in werkloosheid na downturns
2. Veranderingen in belastingopbrengsten na downturns
3. Bij welke lag (1-12 maanden) de effecten het sterkst zijn
4. Statistische significantie van de effecten
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def set_plot_style():
    """Stel plot style in."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 10


def load_data(project_root: Path) -> pd.DataFrame:
    """Laad de combined dataset."""
    processed_dir = project_root / "data" / "processed"
    df = pd.read_csv(processed_dir / "combined_full_dataset.csv", 
                     index_col='date', parse_dates=True)
    return df


def calculate_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Bereken veranderingen in werkloosheid en belastingopbrengsten."""
    df = df.copy()
    
    # Absolute veranderingen
    df['unemployment_change'] = df['unemployment_rate'].diff()
    df['tax_change'] = df['federal_income_tax_revenue'].diff()
    
    # Percentage veranderingen
    df['unemployment_pct_change'] = df['unemployment_rate'].pct_change() * 100
    df['tax_pct_change'] = df['federal_income_tax_revenue'].pct_change() * 100
    
    return df


def analyze_lag_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyseer correlatie tussen downturn lags en veranderingen in 
    werkloosheid en belastingopbrengsten.
    """
    results = []
    
    for lag in range(1, 13):
        lag_col = f'downturn_lag_{lag}m'
        
        # Correlaties met werkloosheid
        corr_unemp_abs = df[lag_col].corr(df['unemployment_change'])
        corr_unemp_pct = df[lag_col].corr(df['unemployment_pct_change'])
        
        # Correlaties met belastingen
        corr_tax_abs = df[lag_col].corr(df['tax_change'])
        corr_tax_pct = df[lag_col].corr(df['tax_pct_change'])
        
        # Gemiddelde verandering wanneer downturn_lag = 1
        downturn_mask = df[lag_col] == 1
        if downturn_mask.sum() > 0:
            avg_unemp_change = df.loc[downturn_mask, 'unemployment_change'].mean()
            avg_tax_change = df.loc[downturn_mask, 'tax_change'].mean()
            avg_unemp_pct = df.loc[downturn_mask, 'unemployment_pct_change'].mean()
            avg_tax_pct = df.loc[downturn_mask, 'tax_pct_change'].mean()
        else:
            avg_unemp_change = avg_tax_change = avg_unemp_pct = avg_tax_pct = np.nan
        
        results.append({
            'lag_months': lag,
            'corr_unemployment_abs': corr_unemp_abs,
            'corr_unemployment_pct': corr_unemp_pct,
            'corr_tax_abs': corr_tax_abs,
            'corr_tax_pct': corr_tax_pct,
            'avg_unemployment_change': avg_unemp_change,
            'avg_tax_change': avg_tax_change,
            'avg_unemployment_pct_change': avg_unemp_pct,
            'avg_tax_pct_change': avg_tax_pct,
        })
    
    return pd.DataFrame(results)


def perform_statistical_tests(df: pd.DataFrame) -> dict:
    """
    Voer statistische tests uit om te bepalen of de effecten 
    significant zijn.
    """
    results = {}
    
    for lag in range(1, 13):
        lag_col = f'downturn_lag_{lag}m'
        
        # Split data: met en zonder downturn X maanden geleden
        with_downturn = df[df[lag_col] == 1]['unemployment_change'].dropna()
        without_downturn = df[df[lag_col] == 0]['unemployment_change'].dropna()
        
        # T-test voor werkloosheid
        if len(with_downturn) > 1 and len(without_downturn) > 1:
            t_stat_unemp, p_val_unemp = stats.ttest_ind(with_downturn, without_downturn)
        else:
            t_stat_unemp, p_val_unemp = np.nan, np.nan
        
        # T-test voor belastingen
        with_downturn_tax = df[df[lag_col] == 1]['tax_change'].dropna()
        without_downturn_tax = df[df[lag_col] == 0]['tax_change'].dropna()
        
        if len(with_downturn_tax) > 1 and len(without_downturn_tax) > 1:
            t_stat_tax, p_val_tax = stats.ttest_ind(with_downturn_tax, without_downturn_tax)
        else:
            t_stat_tax, p_val_tax = np.nan, np.nan
        
        results[lag] = {
            'unemployment_t_stat': t_stat_unemp,
            'unemployment_p_value': p_val_unemp,
            'tax_t_stat': t_stat_tax,
            'tax_p_value': p_val_tax,
        }
    
    return results


def plot_correlation_analysis(corr_results: pd.DataFrame, output_dir: Path):
    """Plot correlaties tussen lags en veranderingen."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Correlatie Analyse: Downturn Lags vs Veranderingen', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    lags = corr_results['lag_months']
    
    # Plot 1: Correlatie met werkloosheid verandering
    axes[0, 0].bar(lags, corr_results['corr_unemployment_abs'], 
                   color='darkred', alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 0].set_xlabel('Maanden na Downturn', fontweight='bold')
    axes[0, 0].set_ylabel('Correlatie Coëfficiënt', fontweight='bold')
    axes[0, 0].set_title('Correlatie: Downturn Lag → Werkloosheid Verandering (abs)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_xticks(lags)
    
    # Plot 2: Correlatie met belastingen verandering
    axes[0, 1].bar(lags, corr_results['corr_tax_abs'], 
                   color='darkgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_xlabel('Maanden na Downturn', fontweight='bold')
    axes[0, 1].set_ylabel('Correlatie Coëfficiënt', fontweight='bold')
    axes[0, 1].set_title('Correlatie: Downturn Lag → Belastingopbrengsten Verandering (abs)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_xticks(lags)
    
    # Plot 3: Gemiddelde werkloosheid verandering
    axes[1, 0].plot(lags, corr_results['avg_unemployment_change'], 
                    marker='o', linewidth=2, markersize=8, color='darkred', label='Abs. verandering')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Maanden na Downturn', fontweight='bold')
    axes[1, 0].set_ylabel('Gem. Verandering Werkloosheid (%-punt)', fontweight='bold')
    axes[1, 0].set_title('Gemiddelde Werkloosheid Verandering na Downturn')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(lags)
    axes[1, 0].legend()
    
    # Plot 4: Gemiddelde belasting verandering
    axes[1, 1].plot(lags, corr_results['avg_tax_change'], 
                    marker='o', linewidth=2, markersize=8, color='darkgreen', label='Abs. verandering')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Maanden na Downturn', fontweight='bold')
    axes[1, 1].set_ylabel('Gem. Verandering Belastingen (mld $)', fontweight='bold')
    axes[1, 1].set_title('Gemiddelde Belastingopbrengsten Verandering na Downturn')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(lags)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lagged_effects_correlation.png', dpi=300, bbox_inches='tight')
    print(f"✅ Opgeslagen: lagged_effects_correlation.png")
    plt.close()


def plot_heatmap_analysis(corr_results: pd.DataFrame, output_dir: Path):
    """Maak een heatmap van de correlaties."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap
    data = corr_results[['lag_months', 'corr_unemployment_abs', 'corr_tax_abs']].set_index('lag_months').T
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Correlatie Coëfficiënt'},
                linewidths=1, linecolor='black', ax=ax)
    
    ax.set_xlabel('Maanden na Downturn', fontweight='bold')
    ax.set_ylabel('Variabele', fontweight='bold')
    ax.set_yticklabels(['Werkloosheid', 'Belastingopbrengsten'], rotation=0)
    ax.set_title('Heatmap: Correlatie tussen Downturn Lags en Veranderingen',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lagged_effects_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Opgeslagen: lagged_effects_heatmap.png")
    plt.close()


def create_results_summary(corr_results: pd.DataFrame, stat_tests: dict, output_dir: Path):
    """Maak een samenvatting van de resultaten."""
    
    # Find peak lags
    peak_unemp_lag = corr_results.loc[corr_results['avg_unemployment_change'].idxmax(), 'lag_months']
    peak_tax_lag = corr_results.loc[corr_results['avg_tax_change'].abs().idxmax(), 'lag_months']
    
    peak_unemp_corr_lag = corr_results.loc[corr_results['corr_unemployment_abs'].idxmax(), 'lag_months']
    peak_tax_corr_lag = corr_results.loc[corr_results['corr_tax_abs'].abs().idxmax(), 'lag_months']
    
    summary = f"""
{'=' * 80}
RESULTATEN: LAGGED EFFECTS ANALYSE
{'=' * 80}

ONDERZOEKSVRAAG:
How long after a U.S. stock market downturn do unemployment and federal 
income tax revenues change?

{'=' * 80}
BELANGRIJKSTE BEVINDINGEN:
{'=' * 80}

1. WERKLOOSHEID:
   - Sterkste gemiddelde stijging: {peak_unemp_lag} maanden na downturn
   - Gemiddelde verandering op dat moment: {corr_results.loc[corr_results['lag_months'] == peak_unemp_lag, 'avg_unemployment_change'].values[0]:.3f} %-punt
   - Sterkste correlatie bij: {peak_unemp_corr_lag} maanden
   - Correlatie waarde: {corr_results.loc[corr_results['lag_months'] == peak_unemp_corr_lag, 'corr_unemployment_abs'].values[0]:.3f}

2. BELASTINGOPBRENGSTEN:
   - Sterkste gemiddelde verandering: {peak_tax_lag} maanden na downturn
   - Gemiddelde verandering op dat moment: {corr_results.loc[corr_results['lag_months'] == peak_tax_lag, 'avg_tax_change'].values[0]:.2f} miljard $
   - Sterkste correlatie bij: {peak_tax_corr_lag} maanden
   - Correlatie waarde: {corr_results.loc[corr_results['lag_months'] == peak_tax_corr_lag, 'corr_tax_abs'].values[0]:.3f}

{'=' * 80}
CORRELATIE TABEL (per lag):
{'=' * 80}

"""
    
    # Add correlation table
    summary += corr_results.to_string(index=False)
    summary += "\n\n"
    
    # Add statistical significance
    summary += f"""
{'=' * 80}
STATISTISCHE SIGNIFICANTIE (p-values < 0.05 zijn significant):
{'=' * 80}

Lag | Werkloosheid p-value | Belastingen p-value | Significant?
----|---------------------|---------------------|-------------
"""
    
    for lag, test_result in stat_tests.items():
        p_unemp = test_result['unemployment_p_value']
        p_tax = test_result['tax_p_value']
        sig_unemp = '✓' if p_unemp < 0.05 else '✗'
        sig_tax = '✓' if p_tax < 0.05 else '✗'
        summary += f"{lag:3d} | {p_unemp:19.4f} | {p_tax:19.4f} | Unemp:{sig_unemp} Tax:{sig_tax}\n"
    
    summary += f"""
{'=' * 80}
CONCLUSIE:
{'=' * 80}

Na een stock market downturn (≥5% maandelijkse daling):

• WERKLOOSHEID stijgt het meest na ongeveer {peak_unemp_lag} maanden
• BELASTINGOPBRENGSTEN veranderen het meest na ongeveer {peak_tax_lag} maanden

Dit suggereert een vertraagd effect waarbij economische indicatoren
pas enkele maanden na een marktcrash significant veranderen.

{'=' * 80}
"""
    
    # Save summary
    with open(output_dir / 'analysis_results.txt', 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"✅ Opgeslagen: analysis_results.txt")
    
    # Also save as CSV
    corr_results.to_csv(output_dir / 'correlation_results.csv', index=False)
    print(f"✅ Opgeslagen: correlation_results.csv")


def main():
    """Hoofdfunctie."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "figures"
    tables_dir = project_root / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LAGGED EFFECTS ANALYSE")
    print("=" * 80)
    print()
    
    set_plot_style()
    
    print("📊 Data laden...")
    df = load_data(project_root)
    print(f"   Geladen: {len(df)} observaties")
    print()
    
    print("📈 Veranderingen berekenen...")
    df = calculate_changes(df)
    print("   ✅ Unemployment changes berekend")
    print("   ✅ Tax revenue changes berekend")
    print()
    
    print("🔍 Correlatie analyse uitvoeren...")
    corr_results = analyze_lag_correlations(df)
    print("   ✅ Correlaties berekend voor alle lags (1-12 maanden)")
    print()
    
    print("📊 Statistische tests uitvoeren...")
    stat_tests = perform_statistical_tests(df)
    print("   ✅ T-tests uitgevoerd")
    print()
    
    print("📉 Visualisaties maken...")
    plot_correlation_analysis(corr_results, output_dir)
    plot_heatmap_analysis(corr_results, output_dir)
    print()
    
    print("📝 Resultaten samenvatten...")
    create_results_summary(corr_results, stat_tests, tables_dir)
    print()
    
    print("=" * 80)
    print("✅ ANALYSE VOLTOOID!")
    print("=" * 80)
    print()
    print("📁 Outputs:")
    print(f"   - Figuren: {output_dir}/")
    print(f"   - Tabellen: {tables_dir}/")


if __name__ == "__main__":
    main()
