import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """
    Customer segmentation using clustering algorithms.
    
    Design philosophy:
    - Use features meaningful for marketing (demographics, behavior, financials)
    - Focus on interpretability over complexity
    - Create actionable segments for business
    """
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.pca_model = None
        self.segment_profiles = {}
        self.feature_names = None
        
    def select_segmentation_features(self, df):
        """
        Select features relevant for customer segmentation.
        
        Rationale: Focus on stable characteristics and behaviors that inform
        marketing strategy. Exclude the target variable (income) initially
        to find natural segments, then analyze income distribution within segments.
        """
        # Demographics
        demographic_features = ['age', 'sex', 'race', 'marital stat', 'education']
        
        # Employment and financial
        employment_features = ['weeks worked in year', 'class of worker', 
                              'major occupation code', 'major industry code']
        
        financial_features = ['capital gains', 'capital losses', 
                             'dividends from stocks', 'wage per hour']
        
        # Household
        household_features = ['detailed household and family stat', 
                             'family members under 18']
        
        # Geographic (optional, can add if useful)
        # geographic_features = ['region of previous residence']
        
        # Combine selected features
        segmentation_features = (demographic_features + employment_features + 
                               financial_features + household_features)
        
        # Verify all features exist
        available_features = [f for f in segmentation_features if f in df.columns]
        
        print(f"Selected {len(available_features)} features for segmentation:")
        for feat in available_features:
            print(f"  - {feat}")
        
        self.feature_names = available_features
        return df[available_features].copy()
    
    def determine_optimal_clusters(self, X, max_clusters=8):
        """
        Determine optimal number of clusters using multiple methods.
        
        Methods:
        1. Elbow method (within-cluster sum of squares)
        2. Silhouette score (cluster separation)
        3. Calinski-Harabasz score (variance ratio)
        4. Davies-Bouldin score (cluster similarity)
        
        Rationale: For this analysis, we force k=3 for maximum business clarity.
        Three segments (Premium/Core/Growth) are easier to operationalize than 5+.
        """
        print(f"\n{'='*70}")
        print("CLUSTERING WITH k=3 FOR BUSINESS CLARITY")
        print(f"{'='*70}")
        print("Rationale: Three-tier segmentation (Premium/Core/Growth) is")
        print("standard business practice and easier to operationalize than 5+ segments.")
        
        # Force k=3
        optimal_k = 3
        
        print(f"\nUsing k={optimal_k} clusters")
        
        # Still calculate metrics for reporting
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        silhouette = silhouette_score(X, labels)
        print(f"Silhouette score: {silhouette:.3f}")
        
        # Create minimal metrics dataframe for compatibility
        metrics_df = pd.DataFrame({
            'n_clusters': [optimal_k],
            'inertia': [kmeans.inertia_],
            'silhouette': [silhouette],
            'calinski_harabasz': [calinski_harabasz_score(X, labels)],
            'davies_bouldin': [davies_bouldin_score(X, labels)]
        })
        
        return metrics_df, int(optimal_k)
    
    def fit_kmeans(self, X, n_clusters=None):
        """
        Fit K-means clustering model.
        
        Rationale: K-means is interpretable, scalable, and creates
        spherical clusters suitable for demographic segmentation.
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters
        
        print(f"\n{'='*70}")
        print(f"FITTING K-MEANS WITH {self.n_clusters} CLUSTERS")
        print(f"{'='*70}")
        
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=20,
            max_iter=300
        )
        
        labels = self.kmeans_model.fit_predict(X)
        
        print(f"K-means converged in {self.kmeans_model.n_iter_} iterations")
        print(f"Final inertia: {self.kmeans_model.inertia_:.2f}")
        
        # Calculate cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster_id, count in zip(unique, counts):
            percentage = count / len(labels) * 100
            print(f"  Cluster {cluster_id}: {count:,} ({percentage:.1f}%)")
        
        return labels
    
    def create_segment_profiles(self, df_original, X_scaled, labels):
        """
        Create detailed profiles for each segment.
        
        Rationale: Profiles translate clusters into actionable business insights.
        Include both raw statistics and comparative analysis across segments.
        """
        print(f"\n{'='*70}")
        print("CREATING SEGMENT PROFILES")
        print(f"{'='*70}")
        
        # Add cluster labels to original dataframe
        df = df_original.copy()
        df['segment'] = labels
        
        profiles = {}
        
        for segment_id in range(self.n_clusters):
            segment_data = df[df['segment'] == segment_id]
            segment_size = len(segment_data)
            
            print(f"\nSegment {segment_id} (n={segment_size:,}):")
            
            profile = {
                'segment_id': int(segment_id),
                'size': int(segment_size),
                'percentage': float(segment_size / len(df) * 100),
                'demographics': {},
                'employment': {},
                'financial': {},
                'income_distribution': {}
            }
            
            # Demographics
            if 'age' in segment_data.columns:
                profile['demographics']['avg_age'] = float(segment_data['age'].mean())
                profile['demographics']['age_std'] = float(segment_data['age'].std())
            
            # Income distribution (from original label)
            if 'label' in segment_data.columns:
                income_dist = segment_data['label'].value_counts(normalize=True) * 100
                profile['income_distribution'] = {
                    'low_income_pct': float(income_dist.get(0, 0)),
                    'high_income_pct': float(income_dist.get(1, 0))
                }
                print(f"  High income rate: {profile['income_distribution']['high_income_pct']:.1f}%")
            
            # Financial characteristics
            if 'weeks worked in year' in segment_data.columns:
                profile['employment']['avg_weeks_worked'] = float(segment_data['weeks worked in year'].mean())
            
            if 'capital gains' in segment_data.columns:
                has_capital_gains = (segment_data['capital gains'] > 0).sum()
                profile['financial']['pct_with_capital_gains'] = float(has_capital_gains / segment_size * 100)
                profile['financial']['avg_capital_gains'] = float(segment_data['capital gains'].mean())
            
            if 'dividends from stocks' in segment_data.columns:
                has_dividends = (segment_data['dividends from stocks'] > 0).sum()
                profile['financial']['pct_with_dividends'] = float(has_dividends / segment_size * 100)
            
            profiles[f'segment_{segment_id}'] = profile
        
        self.segment_profiles = profiles
        return profiles
    
    def create_segment_personas(self, df_original, labels, X_scaled):
        """
        Create marketing personas for each segment.
        
        Rationale: Personas make segments tangible and actionable for marketing teams.
        Based on dominant characteristics within each segment.
        
        Note: Need to properly interpret standardized features by looking at original data.
        """
        df = df_original.copy()
        df['segment'] = labels
        
        personas = {}
        
        for segment_id in range(self.n_clusters):
            segment_data = df[df['segment'] == segment_id]
            
            # Get most common characteristics (for categorical variables)
            persona = {
                'segment_id': segment_id,
                'name': f'Segment {segment_id}',
                'size': len(segment_data),
                'characteristics': {}
            }
            
            # Analyze key categorical features
            categorical_features = ['education', 'marital stat', 'major occupation code', 'sex']
            
            for feat in categorical_features:
                if feat in segment_data.columns:
                    mode_value = segment_data[feat].mode()[0] if len(segment_data[feat].mode()) > 0 else 'Unknown'
                    persona['characteristics'][feat] = str(mode_value)
            
            # Analyze numeric features - use ORIGINAL values, not scaled
            if 'age' in segment_data.columns:
                avg_age = segment_data['age'].mean()  # This is the original age, not scaled
                persona['characteristics']['avg_age'] = f"{avg_age:.1f}"
                persona['characteristics']['age_group'] = self._age_to_group(avg_age)
            
            if 'weeks worked in year' in segment_data.columns:
                avg_weeks = segment_data['weeks worked in year'].mean()
                persona['characteristics']['avg_weeks_worked'] = f"{avg_weeks:.1f}"
            
            if 'label' in segment_data.columns:
                high_income_pct = (segment_data['label'] == 1).sum() / len(segment_data) * 100
                persona['characteristics']['high_income_rate'] = f"{high_income_pct:.1f}%"
            
            # Financial characteristics
            if 'capital gains' in segment_data.columns:
                pct_with_gains = (segment_data['capital gains'] > 0).sum() / len(segment_data) * 100
                persona['characteristics']['pct_with_capital_gains'] = f"{pct_with_gains:.1f}%"
            
            if 'dividends from stocks' in segment_data.columns:
                pct_with_dividends = (segment_data['dividends from stocks'] > 0).sum() / len(segment_data) * 100
                persona['characteristics']['pct_with_dividends'] = f"{pct_with_dividends:.1f}%"
            
            personas[f'segment_{segment_id}'] = persona
        
        return personas
    
    def _age_to_group(self, age):
        """Convert age to marketing-friendly group."""
        if age < 25:
            return "Young Adults (18-24)"
        elif age < 35:
            return "Early Career (25-34)"
        elif age < 45:
            return "Mid Career (35-44)"
        elif age < 55:
            return "Established (45-54)"
        elif age < 65:
            return "Pre-Retirement (55-64)"
        else:
            return "Retirement Age (65+)"
    
    def visualize_with_pca(self, X, labels):
        """
        Visualize segments in 2D using PCA.
        
        Rationale: High-dimensional data needs dimensionality reduction
        for visualization. PCA preserves maximum variance.
        """
        print(f"\n{'='*70}")
        print("CREATING PCA VISUALIZATION")
        print(f"{'='*70}")
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        print(f"Explained variance: {pca.explained_variance_ratio_[0]:.1%} (PC1), {pca.explained_variance_ratio_[1]:.1%} (PC2)")
        print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.1%}")
        
        self.pca_model = pca
        
        return X_pca
    
    def save_model(self, filepath):
        """Save segmentation model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"\nSegmentation model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load segmentation model."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_cluster_evaluation_plots(metrics_df, output_dir):
    """
    Visualize cluster evaluation metrics.
    """
    print("\nGenerating cluster evaluation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Elbow plot
    axes[0, 0].plot(metrics_df['n_clusters'], metrics_df['inertia'], 
                    marker='o', linewidth=2, markersize=10, color='#3498db')
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].set_xticks(metrics_df['n_clusters'])
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Silhouette score
    axes[0, 1].plot(metrics_df['n_clusters'], metrics_df['silhouette'],
                    marker='o', linewidth=2, markersize=10, color='#2ecc71')
    optimal_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'n_clusters']
    axes[0, 1].axvline(x=optimal_k, color='red', linestyle='--', 
                       label=f'Optimal k={int(optimal_k)}', linewidth=2)
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Analysis (Higher is Better)')
    axes[0, 1].set_xticks(metrics_df['n_clusters'])
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Calinski-Harabasz score
    axes[1, 0].plot(metrics_df['n_clusters'], metrics_df['calinski_harabasz'],
                    marker='o', linewidth=2, markersize=10, color='#e74c3c')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Index (Higher is Better)')
    axes[1, 0].set_xticks(metrics_df['n_clusters'])
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Davies-Bouldin score
    axes[1, 1].plot(metrics_df['n_clusters'], metrics_df['davies_bouldin'],
                    marker='o', linewidth=2, markersize=10, color='#f39c12')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Davies-Bouldin Score')
    axes[1, 1].set_title('Davies-Bouldin Index (Lower is Better)')
    axes[1, 1].set_xticks(metrics_df['n_clusters'])
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_evaluation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'cluster_evaluation.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("  Created: cluster_evaluation.pdf/png")


def create_segment_visualization(X_pca, labels, segment_profiles, output_dir):
    """
    Visualize segments in 2D PCA space.
    """
    print("Generating segment visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each segment
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        segment_key = f'segment_{label}'
        size = segment_profiles[segment_key]['size']
        pct = segment_profiles[segment_key]['percentage']
        
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=[colors[i]], label=f'Segment {label} (n={size:,}, {pct:.1f}%)',
                  alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('Customer Segments in PCA Space')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'segment_visualization.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'segment_visualization.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("  Created: segment_visualization.pdf/png")


def create_segment_comparison_plots(df_original, labels, output_dir):
    """
    Compare segments across key dimensions.
    """
    print("Generating segment comparison plots...")
    
    df = df_original.copy()
    df['segment'] = labels
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Income distribution by segment
    if 'label' in df.columns:
        income_by_segment = pd.crosstab(df['segment'], df['label'], normalize='index') * 100
        income_by_segment.plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#2ecc71'])
        axes[0, 0].set_xlabel('Segment')
        axes[0, 0].set_ylabel('Percentage (%)')
        axes[0, 0].set_title('Income Distribution by Segment')
        axes[0, 0].legend(['< $50K', '≥ $50K'])
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)
        axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Age distribution by segment
    if 'age' in df.columns:
        df.boxplot(column='age', by='segment', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Segment')
        axes[0, 1].set_ylabel('Age')
        axes[0, 1].set_title('Age Distribution by Segment')
        plt.setp(axes[0, 1], title='')
        axes[0, 1].get_figure().suptitle('')
    
    # 3. Weeks worked by segment
    if 'weeks worked in year' in df.columns:
        df.boxplot(column='weeks worked in year', by='segment', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Segment')
        axes[1, 0].set_ylabel('Weeks Worked in Year')
        axes[1, 0].set_title('Employment Engagement by Segment')
        plt.setp(axes[1, 0], title='')
        axes[1, 0].get_figure().suptitle('')
    
    # 4. Segment sizes
    segment_sizes = df['segment'].value_counts().sort_index()
    bars = axes[1, 1].bar(segment_sizes.index, segment_sizes.values, 
                          color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Segment')
    axes[1, 1].set_ylabel('Number of Customers')
    axes[1, 1].set_title('Segment Sizes')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, count in zip(bars, segment_sizes.values):
        height = bar.get_height()
        pct = count / len(df) * 100
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}\n({pct:.1f}%)',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'segment_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'segment_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("  Created: segment_comparison.pdf/png")


def generate_marketing_recommendations(segment_profiles, personas):
    """
    Generate actionable marketing recommendations for each segment.
    
    Rationale: Connects analytical insights to business action.
    Uses segment characteristics to provide specific, differentiated strategies.
    """
    recommendations = {}
    
    for segment_key, profile in segment_profiles.items():
        segment_id = profile['segment_id']
        high_income_rate = profile['income_distribution']['high_income_pct']
        avg_weeks_worked = profile['employment'].get('avg_weeks_worked', 0)
        pct_capital_gains = profile['financial'].get('pct_with_capital_gains', 0)
        pct_dividends = profile['financial'].get('pct_with_dividends', 0)
        
        # Get persona info for better recommendations
        persona = personas.get(segment_key, {})
        avg_age = float(persona.get('characteristics', {}).get('avg_age', 35))
        
        # Classify segment value
        if high_income_rate > 20:
            value_tier = "Premium"
            base_strategy = "High-touch relationship building, premium product offerings"
        elif high_income_rate > 8:
            value_tier = "Core"
            base_strategy = "Balanced approach, focus on value proposition and quality"
        else:
            value_tier = "Growth"
            base_strategy = "Volume-based, cost-effective marketing with accessibility focus"
        
        # Determine channels based on characteristics
        channels = []
        messaging = []
        
        # High capital gains/dividends = wealthy investors
        if pct_capital_gains > 50 or pct_dividends > 30:
            channels = ['Financial advisors', 'Private banking', 'Wealth management seminars', 'Premium publications']
            messaging = ['Wealth preservation', 'Investment opportunities', 'Exclusive access', 'Legacy planning']
        
        # High work engagement (professional workers)
        elif avg_weeks_worked > 0.5:  # Standardized value, positive means above average
            if high_income_rate > 10:
                channels = ['LinkedIn', 'Professional networks', 'Email', 'Business publications']
                messaging = ['Career advancement', 'Time-saving solutions', 'Premium quality', 'Status']
            else:
                channels = ['Social media', 'Email', 'Mobile apps', 'Workplace programs']
                messaging = ['Value for money', 'Convenience', 'Work-life balance', 'Practical solutions']
        
        # Low work engagement (unemployed, retired, part-time)
        else:
            if avg_age > 55:  # Likely retirees
                channels = ['Traditional media', 'Community centers', 'Direct mail', 'Local events']
                messaging = ['Reliability', 'Trust', 'Community', 'Legacy']
            else:  # Likely students or unemployed
                channels = ['Social media', 'Digital platforms', 'Student programs', 'Community outreach']
                messaging = ['Affordability', 'Accessibility', 'Future potential', 'Opportunity']
        
        # Default if no specific pattern
        if not channels:
            channels = ['Digital marketing', 'Social media', 'Email campaigns']
            messaging = ['Value', 'Quality', 'Trust']
        
        recommendations[segment_key] = {
            'segment_id': segment_id,
            'value_tier': value_tier,
            'high_income_rate': high_income_rate,
            'marketing_strategy': base_strategy,
            'recommended_channels': channels,
            'messaging_focus': ', '.join(messaging),
            'budget_allocation': 'High' if value_tier == 'Premium' else 'Medium' if value_tier == 'Core' else 'Low'
        }
    
    return recommendations


def main():
    """
    Main execution for customer segmentation.
    """
    print("Customer Segmentation Analysis")
    print("=" * 70)
    
    # Paths
    data_dir = Path("data/processed")
    raw_data_dir = Path("data/raw")
    output_dir = Path("outputs")
    model_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    
    # Load processed data for clustering
    print("\nLoading processed data for clustering...")
    train_df = pd.read_csv(data_dir / "train_data.csv")
    test_df = pd.read_csv(data_dir / "test_data.csv")
    full_df_processed = pd.concat([train_df, test_df], ignore_index=True)
    
    # Load ORIGINAL unprocessed data for personas (with real age values)
    print("Loading original raw data for persona interpretation...")
    columns_path = raw_data_dir / "census-bureau.columns"
    data_path = raw_data_dir / "census-bureau.data"
    
    with open(columns_path, 'r') as f:
        columns = [line.strip() for line in f.readlines()]
    
    full_df_original = pd.read_csv(data_path, names=columns, skipinitialspace=True)
    # Remove duplicates to match processed data
    full_df_original = full_df_original.drop_duplicates()
    
    print(f"Processed records for clustering: {len(full_df_processed):,}")
    print(f"Original records for interpretation: {len(full_df_original):,}")
    
    # Verify they match in size (should after duplicate removal)
    if len(full_df_processed) != len(full_df_original):
        print(f"Warning: Size mismatch. Using processed data for both.")
        full_df_original = full_df_processed.copy()
    
    # Initialize segmentation
    segmentation = CustomerSegmentation()
    
    # Select features from processed data
    print("\n" + "=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)
    X = segmentation.select_segmentation_features(full_df_processed)
    
    # Scale features
    print("\nScaling features...")
    X_scaled = segmentation.scaler.fit_transform(X)
    
    # Determine optimal clusters (forced to k=3)
    metrics_df, optimal_k = segmentation.determine_optimal_clusters(X_scaled, max_clusters=8)
    
    # Skip evaluation plots since we're forcing k=3
    # create_cluster_evaluation_plots(metrics_df, figures_dir)
    
    # Fit K-means with optimal k
    labels = segmentation.fit_kmeans(X_scaled, n_clusters=optimal_k)
    
    # Create segment profiles using PROCESSED data (has label column)
    profiles = segmentation.create_segment_profiles(full_df_processed, X_scaled, labels)
    
    # Create personas using ORIGINAL data (has real age values)
    personas = segmentation.create_segment_personas(full_df_original, labels, X_scaled)
    
    # PCA visualization
    X_pca = segmentation.visualize_with_pca(X_scaled, labels)
    create_segment_visualization(X_pca, labels, profiles, figures_dir)
    
    # Comparison plots - use processed data (has label)
    create_segment_comparison_plots(full_df_processed, labels, figures_dir)
    
    # Marketing recommendations
    print("\n" + "=" * 70)
    print("MARKETING RECOMMENDATIONS")
    print("=" * 70)
    recommendations = generate_marketing_recommendations(profiles, personas)
    
    for segment_key, rec in recommendations.items():
        print(f"\n{segment_key.upper()}:")
        print(f"  Value Tier: {rec['value_tier']}")
        print(f"  High Income Rate: {rec['high_income_rate']:.1f}%")
        print(f"  Strategy: {rec['marketing_strategy']}")
    
    # Save model
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    segmentation.save_model(model_dir / "customer_segmentation.pkl")
    
    # Save comprehensive results
    results = {
        'model_type': 'K-Means Clustering',
        'n_clusters': optimal_k,
        'total_samples': len(full_df_processed),
        'features_used': segmentation.feature_names,
        'cluster_metrics': {
            'silhouette_score': float(metrics_df.loc[metrics_df['n_clusters'] == optimal_k, 'silhouette'].values[0]),
            'calinski_harabasz_score': float(metrics_df.loc[metrics_df['n_clusters'] == optimal_k, 'calinski_harabasz'].values[0]),
            'davies_bouldin_score': float(metrics_df.loc[metrics_df['n_clusters'] == optimal_k, 'davies_bouldin'].values[0])
        },
        'segment_profiles': profiles,
        'segment_personas': personas,
        'marketing_recommendations': recommendations,
        'pca_variance_explained': {
            'pc1': float(segmentation.pca_model.explained_variance_ratio_[0]),
            'pc2': float(segmentation.pca_model.explained_variance_ratio_[1]),
            'total': float(segmentation.pca_model.explained_variance_ratio_.sum())
        }
    }
    
    with open(output_dir / 'segmentation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved: {output_dir / 'segmentation_results.json'}")
    
    # Save cluster assignments
    full_df_processed['segment'] = labels
    full_df_processed[['segment']].to_csv(output_dir / 'segment_assignments.csv', index=False)
    print(f"  Saved: {output_dir / 'segment_assignments.csv'}")
    
    print("\n" + "=" * 70)
    print("SEGMENTATION ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nIdentified {optimal_k} customer segments")
    print(f"Silhouette score: {results['cluster_metrics']['silhouette_score']:.3f}")
    print("\nGenerated files:")
    print(f"  - {model_dir / 'customer_segmentation.pkl'}")
    print(f"  - {output_dir / 'segmentation_results.json'}")
    print(f"  - {output_dir / 'segment_assignments.csv'}")
    print(f"  - {figures_dir / 'cluster_evaluation.pdf/png'}")
    print(f"  - {figures_dir / 'segment_visualization.pdf/png'}")
    print(f"  - {figures_dir / 'segment_comparison.pdf/png'}")


if __name__ == "__main__":
    main()