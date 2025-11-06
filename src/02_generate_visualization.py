import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set publication quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

sns.set_palette("husl")


def load_data():
    """Load census data and column names."""
    data_path = Path("data/raw/census-bureau.data")
    columns_path = Path("data/raw/census-bureau.columns")
    
    with open(columns_path, 'r') as f:
        columns = [line.strip() for line in f.readlines()]
    
    df = pd.read_csv(data_path, names=columns, skipinitialspace=True)
    return df


def plot_target_distribution(df, output_dir):
    """
    Figure 1: Target variable distribution showing severe class imbalance.
    
    Rationale: Visualizing class imbalance is critical for justifying
    sampling strategies and evaluation metric selection.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Count plot
    target_counts = df['label'].value_counts()
    bars = ax1.bar(range(len(target_counts)), target_counts.values, 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(target_counts)))
    ax1.set_xticklabels(['< $50K', '≥ $50K'])
    ax1.set_ylabel('Number of Records')
    ax1.set_title('Income Distribution (Count)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, target_counts.values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    # Percentage plot
    percentages = (target_counts / len(df) * 100).values
    bars2 = ax2.bar(range(len(target_counts)), percentages,
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(target_counts)))
    ax2.set_xticklabels(['< $50K', '≥ $50K'])
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Income Distribution (Percentage)')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bar, pct in zip(bars2, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_target_distribution.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_target_distribution.png', bbox_inches='tight')
    plt.close()
    print("Created: fig1_target_distribution")


def plot_missing_data_analysis(df, output_dir):
    """
    Figure 2: Missing data patterns across features.
    
    Rationale: Understanding which features have implicit missing values
    (coded as "Not in universe", "?", etc.) informs imputation strategy.
    """
    # Identify columns with "Not in universe" or "?" patterns
    missing_patterns = {}
    
    for col in df.columns:
        total = len(df)
        if df[col].dtype == 'object':
            niu_count = df[col].str.contains('Not in universe', na=False).sum()
            question_count = (df[col] == '?').sum()
            explicit_na = df[col].isna().sum()
            
            total_missing = niu_count + question_count + explicit_na
            if total_missing > 0:
                missing_patterns[col] = {
                    'not_in_universe': niu_count,
                    'question_mark': question_count,
                    'explicit_na': explicit_na,
                    'total': total_missing,
                    'percentage': (total_missing / total) * 100
                }
        else:
            # Check for numeric missing indicators
            zero_count = (df[col] == 0).sum()
            explicit_na = df[col].isna().sum()
            
            # Only count zeros as missing for specific columns where 0 means "not applicable"
            if col in ['detailed industry recode', 'detailed occupation recode']:
                total_missing = zero_count + explicit_na
                if total_missing > 0:
                    missing_patterns[col] = {
                        'zero_code': zero_count,
                        'explicit_na': explicit_na,
                        'total': total_missing,
                        'percentage': (total_missing / total) * 100
                    }
    
    # Sort by percentage
    sorted_missing = sorted(missing_patterns.items(), 
                          key=lambda x: x[1]['percentage'], 
                          reverse=True)
    
    # Take top 15 for visualization
    top_missing = sorted_missing[:15]
    
    if top_missing:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cols = [item[0] for item in top_missing]
        percentages = [item[1]['percentage'] for item in top_missing]
        
        bars = ax.barh(range(len(cols)), percentages, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=8)
        ax.set_xlabel('Missing Data Percentage (%)')
        ax.set_title('Features with Implicit Missing Values\n(Not in universe, ?, or code 0)')
        ax.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {pct:.1f}%',
                   ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fig2_missing_patterns.pdf', bbox_inches='tight')
        plt.savefig(output_dir / 'fig2_missing_patterns.png', bbox_inches='tight')
        plt.close()
        print("Created: fig2_missing_patterns")


def plot_numeric_distributions(df, output_dir):
    """
    Figure 3: Key numeric feature distributions.
    
    Rationale: Understanding distributions helps identify skewness,
    outliers, and appropriate transformation strategies.
    """
    numeric_cols = ['age', 'wage per hour', 'capital gains', 'capital losses', 
                   'dividends from stocks', 'weeks worked in year']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Remove zeros for wage, capital gains/losses, dividends (0 means none)
        if col in ['wage per hour', 'capital gains', 'capital losses', 'dividends from stocks']:
            data = df[df[col] > 0][col]
            title_suffix = ' (excluding zeros)'
        else:
            data = df[col]
            title_suffix = ''
        
        # Handle extreme outliers for visualization
        if col == 'wage per hour':
            data = data[data < 2000]  # Remove extreme outliers for visualization
        
        ax.hist(data, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col.capitalize()}{title_suffix}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add summary statistics
        ax.text(0.98, 0.97, f'Mean: {data.mean():.1f}\nMedian: {data.median():.1f}',
               transform=ax.transAxes, fontsize=7,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_numeric_distributions.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_numeric_distributions.png', bbox_inches='tight')
    plt.close()
    print("Created: fig3_numeric_distributions")


def plot_categorical_top_features(df, output_dir):
    """
    Figure 4: Distribution of key categorical features.
    
    Rationale: Understanding categorical distributions helps identify
    dominant categories and rare classes that may need grouping.
    """
    categorical_cols = ['education', 'marital stat', 'race', 'sex']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        
        value_counts = df[col].value_counts()
        
        # Take top 10 for readability
        if len(value_counts) > 10:
            top_values = value_counts.head(10)
            other_count = value_counts.iloc[10:].sum()
            if other_count > 0:
                top_values['Other'] = other_count
        else:
            top_values = value_counts
        
        bars = ax.barh(range(len(top_values)), top_values.values, 
                       color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels([label[:30] for label in top_values.index], fontsize=8)
        ax.set_xlabel('Count')
        ax.set_title(col.capitalize(), fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, top_values.values)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {count:,}',
                   ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_categorical_distributions.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_categorical_distributions.png', bbox_inches='tight')
    plt.close()
    print("Created: fig4_categorical_distributions")


def plot_income_by_key_features(df, output_dir):
    """
    Figure 5: Income distribution across key demographic features.
    
    Rationale: Understanding how income varies by demographics informs
    feature importance and helps identify patterns for segmentation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Education vs Income
    ax1 = axes[0, 0]
    edu_income = pd.crosstab(df['education'], df['label'], normalize='index') * 100
    edu_income = edu_income.sort_values('50000+.', ascending=False).head(10)
    
    edu_income.plot(kind='barh', stacked=False, ax=ax1, 
                    color=['#e74c3c', '#2ecc71'], alpha=0.7)
    ax1.set_xlabel('Percentage (%)')
    ax1.set_ylabel('')
    ax1.set_title('Income Distribution by Education Level')
    ax1.legend(['< $50K', '≥ $50K'], loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    
    # Age vs Income
    ax2 = axes[0, 1]
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ['0-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    
    age_income = pd.crosstab(df['age_group'], df['label'], normalize='index') * 100
    age_income.plot(kind='bar', ax=ax2, color=['#e74c3c', '#2ecc71'], alpha=0.7)
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Income Distribution by Age Group')
    ax2.legend(['< $50K', '≥ $50K'])
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Marital Status vs Income
    ax3 = axes[1, 0]
    marital_income = pd.crosstab(df['marital stat'], df['label'], normalize='index') * 100
    marital_income = marital_income.sort_values('50000+.', ascending=False)
    
    marital_income.plot(kind='barh', ax=ax3, color=['#e74c3c', '#2ecc71'], alpha=0.7)
    ax3.set_xlabel('Percentage (%)')
    ax3.set_ylabel('')
    ax3.set_title('Income Distribution by Marital Status')
    ax3.legend(['< $50K', '≥ $50K'])
    ax3.grid(axis='x', alpha=0.3)
    
    # Sex vs Income
    ax4 = axes[1, 1]
    sex_income = pd.crosstab(df['sex'], df['label'], normalize='index') * 100
    
    x = np.arange(len(sex_income.index))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, sex_income['- 50000.'], width, 
                    label='< $50K', color='#e74c3c', alpha=0.7)
    bars2 = ax4.bar(x + width/2, sex_income['50000+.'], width,
                    label='≥ $50K', color='#2ecc71', alpha=0.7)
    
    ax4.set_xlabel('Sex')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Income Distribution by Sex')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sex_income.index)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_income_by_demographics.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_income_by_demographics.png', bbox_inches='tight')
    plt.close()
    print("Created: fig5_income_by_demographics")


def plot_correlation_heatmap(df, output_dir):
    """
    Figure 6: Correlation matrix for numeric features.
    
    Rationale: Identifying correlated features helps with feature selection
    and understanding multicollinearity issues.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove weight and year (not predictive features)
    numeric_cols = [col for col in numeric_cols if col not in ['weight', 'year']]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, annot_kws={'size': 7})
    
    ax.set_title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_correlation_matrix.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_correlation_matrix.png', bbox_inches='tight')
    plt.close()
    print("Created: fig6_correlation_matrix")


def generate_summary_statistics_table(df, output_dir):
    """
    Generate LaTeX table of summary statistics.
    
    Rationale: Provides comprehensive overview for report appendix.
    """
    numeric_cols = ['age', 'wage per hour', 'capital gains', 'capital losses',
                   'dividends from stocks', 'weeks worked in year']
    
    summary_stats = []
    for col in numeric_cols:
        stats = df[col].describe()
        summary_stats.append({
            'Feature': col,
            'Mean': f"{stats['mean']:.2f}",
            'Std': f"{stats['std']:.2f}",
            'Min': f"{stats['min']:.2f}",
            'Q1': f"{stats['25%']:.2f}",
            'Median': f"{stats['50%']:.2f}",
            'Q3': f"{stats['75%']:.2f}",
            'Max': f"{stats['max']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Generate LaTeX table
    latex_table = summary_df.to_latex(index=False, 
                                      caption='Summary Statistics of Numeric Features',
                                      label='tab:summary_stats',
                                      column_format='l' + 'r' * 7,
                                      escape=False)
    
    with open(output_dir / 'table_summary_stats.tex', 'w') as f:
        f.write(latex_table)
    
    print("Created: table_summary_stats.tex")


def main():
    """Execute all visualization tasks."""
    print("Starting visualization generation...")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df):,} records")
    
    # Generate all figures
    print("\nGenerating figures...")
    print("-" * 50)
    
    plot_target_distribution(df, output_dir)
    plot_missing_data_analysis(df, output_dir)
    plot_numeric_distributions(df, output_dir)
    plot_categorical_top_features(df, output_dir)
    plot_income_by_key_features(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    
    # Generate summary table
    print("\nGenerating summary statistics table...")
    print("-" * 50)
    generate_summary_statistics_table(df, output_dir)
    
    print("\n" + "=" * 50)
    print("Visualization generation complete!")
    print(f"All outputs saved to: {output_dir}")
    print("\nFiles created:")
    print("  - fig1_target_distribution.pdf/png")
    print("  - fig2_missing_patterns.pdf/png")
    print("  - fig3_numeric_distributions.pdf/png")
    print("  - fig4_categorical_distributions.pdf/png")
    print("  - fig5_income_by_demographics.pdf/png")
    print("  - fig6_correlation_matrix.pdf/png")
    print("  - table_summary_stats.tex")


if __name__ == "__main__":
    main()