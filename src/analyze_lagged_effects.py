"""
Analysis Script: Lagged Effects of Stock Market Downturns
Research Question: How long after a U.S. stock market downturn do unemployment 
                   and federal income tax revenues change?

This script analyzes:
1. Changes in unemployment after downturns
2. Changes in tax revenues after downturns
3. At which lag (1-12 months) the effects are strongest
4. Statistical significance of the effects
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def set_plot_style():
    """Set plot style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 10


def load_data(project_root: Path) -> pd.DataFrame:
    """Load the combined dataset."""
    processed_dir = project_root / "data" / "processed"
    df = pd.read_csv(processed_dir / "combined_full_dataset.csv", 
                     index_col='date', parse_dates=True)
    return df


def calculate_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate changes in unemployment and tax revenues."""
    df = df.copy()
    
    # Absolute changes
    df['unemployment_change'] = df['unemployment_rate'].diff()
    df['tax_change'] = df['federal_income_tax_revenue'].diff()
    
    # Percentage changes
    df['unemployment_pct_change'] = df['unemployment_rate'].pct_change() * 100
    df['tax_pct_change'] = df['federal_income_tax_revenue'].pct_change() * 100
    
    return df


def analyze_lag_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlation between downturn lags and changes in 
    unemployment and tax revenues.
    """
    results = []
    
    for lag in range(1, 13):
        lag_col = f'downturn_lag_{lag}m'
        
        # Correlations with unemployment
        corr_unemp_abs = df[lag_col].corr(df['unemployment_change'])
        corr_unemp_pct = df[lag_col].corr(df['unemployment_pct_change'])
        
        # Correlations with taxes
        corr_tax_abs = df[lag_col].corr(df['tax_change'])
        corr_tax_pct = df[lag_col].corr(df['tax_pct_change'])
        
        # Average change when downturn_lag = 1
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
        
        # T-test for unemployment
        if len(with_downturn) > 1 and len(without_downturn) > 1:
            t_stat_unemp, p_val_unemp = stats.ttest_ind(with_downturn, without_downturn)
        else:
            t_stat_unemp, p_val_unemp = np.nan, np.nan
        
        # T-test for taxes
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
    """Plot correlations between lags and changes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Correlation Analysis: Downturn Lags vs Changes', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    lags = corr_results['lag_months']
    
    # Plot 1: Correlation with unemployment change
    axes[0, 0].bar(lags, corr_results['corr_unemployment_abs'], 
                   color='darkred', alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 0].set_xlabel('Months After Downturn', fontweight='bold')
    axes[0, 0].set_ylabel('Correlation Coefficient', fontweight='bold')
    axes[0, 0].set_title('Correlation: Downturn Lag → Unemployment Change (abs)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_xticks(lags)
    
    # Plot 2: Correlation with tax change
    axes[0, 1].bar(lags, corr_results['corr_tax_abs'], 
                   color='darkgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_xlabel('Months After Downturn', fontweight='bold')
    axes[0, 1].set_ylabel('Correlation Coefficient', fontweight='bold')
    axes[0, 1].set_title('Correlation: Downturn Lag → Tax Revenue Change (abs)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_xticks(lags)
    
    # Plot 3: Average unemployment change
    axes[1, 0].plot(lags, corr_results['avg_unemployment_change'], 
                    marker='o', linewidth=2, markersize=8, color='darkred', label='Abs. change')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Months After Downturn', fontweight='bold')
    axes[1, 0].set_ylabel('Avg. Unemployment Change (%-point)', fontweight='bold')
    axes[1, 0].set_title('Average Unemployment Change After Downturn')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(lags)
    axes[1, 0].legend()
    
    # Plot 4: Average tax change
    axes[1, 1].plot(lags, corr_results['avg_tax_change'], 
                    marker='o', linewidth=2, markersize=8, color='darkgreen', label='Abs. change')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Months After Downturn', fontweight='bold')
    axes[1, 1].set_ylabel('Avg. Tax Revenue Change (billion $)', fontweight='bold')
    axes[1, 1].set_title('Average Tax Revenue Change After Downturn')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(lags)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lagged_effects_correlation.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: lagged_effects_correlation.png")
    plt.close()


def plot_heatmap_analysis(corr_results: pd.DataFrame, output_dir: Path):
    """Create a heatmap of the correlations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap
    data = corr_results[['lag_months', 'corr_unemployment_abs', 'corr_tax_abs']].set_index('lag_months').T
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Correlation Coefficient'},
                linewidths=1, linecolor='black', ax=ax)
    
    ax.set_xlabel('Months After Downturn', fontweight='bold')
    ax.set_ylabel('Variable', fontweight='bold')
    ax.set_yticklabels(['Unemployment', 'Tax Revenue'], rotation=0)
    ax.set_title('Heatmap: Correlation Between Downturn Lags and Changes',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lagged_effects_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: lagged_effects_heatmap.png")
    plt.close()


def create_results_summary(corr_results: pd.DataFrame, stat_tests: dict, output_dir: Path):
    """Create a summary of the results."""
    
    # Find peak lags
    peak_unemp_lag = corr_results.loc[corr_results['avg_unemployment_change'].idxmax(), 'lag_months']
    peak_tax_lag = corr_results.loc[corr_results['avg_tax_change'].abs().idxmax(), 'lag_months']
    
    peak_unemp_corr_lag = corr_results.loc[corr_results['corr_unemployment_abs'].idxmax(), 'lag_months']
    peak_tax_corr_lag = corr_results.loc[corr_results['corr_tax_abs'].abs().idxmax(), 'lag_months']
    
    summary = f"""
{'=' * 80}
RESULTS: LAGGED EFFECTS ANALYSIS
{'=' * 80}

RESEARCH QUESTION:
How long after a U.S. stock market downturn do unemployment and federal 
income tax revenues change?

{'=' * 80}
KEY FINDINGS:
{'=' * 80}

1. UNEMPLOYMENT:
   - Strongest average increase: {peak_unemp_lag} months after downturn
   - Average change at that time: {corr_results.loc[corr_results['lag_months'] == peak_unemp_lag, 'avg_unemployment_change'].values[0]:.3f} %-points
   - Strongest correlation at: {peak_unemp_corr_lag} months
   - Correlation value: {corr_results.loc[corr_results['lag_months'] == peak_unemp_corr_lag, 'corr_unemployment_abs'].values[0]:.3f}

2. TAX REVENUE:
   - Strongest average change: {peak_tax_lag} months after downturn
   - Average change at that time: {corr_results.loc[corr_results['lag_months'] == peak_tax_lag, 'avg_tax_change'].values[0]:.2f} billion $
   - Strongest correlation at: {peak_tax_corr_lag} months
   - Correlation value: {corr_results.loc[corr_results['lag_months'] == peak_tax_corr_lag, 'corr_tax_abs'].values[0]:.3f}

{'=' * 80}
CORRELATION TABLE (per lag):
{'=' * 80}

"""
    
    # Add correlation table
    summary += corr_results.to_string(index=False)
    summary += "\n\n"
    
    # Add statistical significance
    summary += f"""
{'=' * 80}
STATISTICAL SIGNIFICANCE (p-values < 0.05 are significant):
{'=' * 80}

Lag | Unemployment p-value | Tax Revenue p-value | Significant?
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
CONCLUSION:
{'=' * 80}

After a stock market downturn (≥5% monthly decline):

• UNEMPLOYMENT increases most after approximately {peak_unemp_lag} months
• TAX REVENUE changes most after approximately {peak_tax_lag} months

This suggests a delayed effect where economic indicators
only change significantly several months after a market crash.

{'=' * 80}
"""
    
    # Save summary
    with open(output_dir / 'analysis_results.txt', 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"✅ Saved: analysis_results.txt")
    
    # Also save as CSV
    corr_results.to_csv(output_dir / 'correlation_results.csv', index=False)
    print(f"✅ Saved: correlation_results.csv")


def main():
    """Main function."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "figures"
    tables_dir = project_root / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LAGGED EFFECTS ANALYSIS")
    print("=" * 80)
    print()
    
    set_plot_style()
    
    print("📊 Loading data...")
    df = load_data(project_root)
    print(f"   Loaded: {len(df)} observations")
    print()
    
    print("📈 Calculating changes...")
    df = calculate_changes(df)
    print("   ✅ Unemployment changes calculated")
    print("   ✅ Tax revenue changes calculated")
    print()
    
    print("🔍 Performing correlation analysis...")
    corr_results = analyze_lag_correlations(df)
    print("   ✅ Correlations calculated for all lags (1-12 months)")
    print()
    
    print("📊 Performing statistical tests...")
    stat_tests = perform_statistical_tests(df)
    print("   ✅ T-tests completed")
    print()
    
    print("📉 Creating visualizations...")
    plot_correlation_analysis(corr_results, output_dir)
    plot_heatmap_analysis(corr_results, output_dir)
    print()
    
    print("📝 Summarizing results...")
    create_results_summary(corr_results, stat_tests, tables_dir)
    print()
    
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("📁 Outputs:")
    print(f"   - Figures: {output_dir}/")
    print(f"   - Tables: {tables_dir}/")


if __name__ == "__main__":
    main()
