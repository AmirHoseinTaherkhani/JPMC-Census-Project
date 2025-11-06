"""
Data Preprocessing and Cleaning Pipeline for Census Income Dataset

Based on EDA findings:
1. Severe class imbalance (15:1 ratio)
2. Missing values encoded as "Not in universe", "?", and special codes
3. Mix of categorical and numeric features requiring different treatment
4. Weighted sampling must be preserved
5. Duplicate records need handling

Author: Senior Data Scientist, JP Morgan
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class CensusDataPreprocessor:
    """
    Preprocessing pipeline for census income data.
    
    Design rationale:
    - Handles missing value imputation based on semantic meaning
    - Preserves sample weights for proper model training
    - Maintains interpretability for business stakeholders
    - Creates reproducible transformation pipeline
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.numeric_scaler = StandardScaler()
        self.feature_names = None
        self.categorical_features = []
        self.numeric_features = []
        self.processed_columns = []
        
    def identify_missing_indicators(self, df):
        """
        Identify and document missing value patterns.
        
        Rationale: Census data uses multiple missing indicators that need
        consistent handling across features.
        """
        missing_map = {}
        
        for col in df.columns:
            indicators = []
            
            if df[col].dtype == 'object':
                # String-based missing indicators
                if 'Not in universe' in df[col].values:
                    indicators.append('Not in universe')
                if '?' in df[col].values:
                    indicators.append('?')
            else:
                # Numeric missing indicators for specific columns
                if col in ['detailed industry recode', 'detailed occupation recode']:
                    indicators.append(0)
                if col == 'wage per hour':
                    # 0 means not applicable, 9999 is outlier/error code
                    indicators.append(9999)
            
            if indicators:
                missing_map[col] = indicators
        
        return missing_map
    
    def handle_missing_values(self, df):
        """
        Replace missing indicators with NaN for consistent handling.
        
        Rationale: Converting all missing patterns to NaN allows
        standard imputation techniques and makes missingness explicit.
        """
        df = df.copy()
        missing_map = self.identify_missing_indicators(df)
        
        for col, indicators in missing_map.items():
            for indicator in indicators:
                if df[col].dtype == 'object':
                    df.loc[df[col] == indicator, col] = np.nan
                else:
                    df.loc[df[col] == indicator, col] = np.nan
        
        return df
    
    def handle_duplicates(self, df):
        """
        Remove duplicate records.
        
        Rationale: 3,229 duplicates (1.6%) could bias model. Since we have
        sample weights, duplicates don't represent intentional oversampling.
        Removal improves data quality without significant information loss.
        """
        initial_count = len(df)
        df = df.drop_duplicates()
        removed_count = initial_count - len(df)
        
        print(f"Removed {removed_count:,} duplicate records ({removed_count/initial_count*100:.2f}%)")
        return df
    
    def create_feature_groups(self, df):
        """
        Categorize features by type for appropriate preprocessing.
        
        Rationale: Different feature types require different transformations.
        This grouping makes the pipeline modular and maintainable.
        """
        # Exclude target, weight, and year from features
        exclude_cols = ['label', 'weight', 'year']
        
        # Identify truly numeric features based on domain knowledge
        # These are continuous or high-cardinality numeric variables
        continuous_numeric = [
            'age', 'wage per hour', 'capital gains', 'capital losses',
            'dividends from stocks', 'weeks worked in year', 
            'num persons worked for employer'
        ]
        
        # Check which continuous numeric features exist in dataframe
        numeric_features = [col for col in continuous_numeric 
                          if col in df.columns and col not in exclude_cols]
        
        # All other features (including numeric-encoded categoricals) are treated as categorical
        categorical_features = [col for col in df.columns 
                              if col not in exclude_cols + numeric_features]
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        print(f"\nFeature categorization:")
        print(f"  Numeric features: {len(numeric_features)}")
        print(f"    {numeric_features}")
        print(f"  Categorical features: {len(categorical_features)}")
        
        return numeric_features, categorical_features
    
    def impute_missing_values(self, df):
        """
        Impute missing values using appropriate strategies.
        
        Rationale:
        - Numeric: median imputation (robust to outliers)
        - Categorical: mode imputation or 'Unknown' category
        - Preserves information while maintaining complete cases
        """
        df = df.copy()
        
        # Numeric imputation
        for col in self.numeric_features:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Imputed {col} with median: {median_val:.2f}")
        
        # Categorical imputation
        for col in self.categorical_features:
            if df[col].isna().any():
                # Use mode if available, otherwise 'Unknown'
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    print(f"  Imputed {col} with mode: {mode_val}")
                else:
                    df[col].fillna('Unknown', inplace=True)
                    print(f"  Imputed {col} with 'Unknown'")
        
        return df
    
    def encode_categorical_variables(self, df, fit=True):
        """
        Encode categorical variables for modeling.
        
        Rationale: Label encoding for tree-based models maintains
        ordinality where present and reduces dimensionality compared
        to one-hot encoding with 40+ features.
        """
        df = df.copy()
        
        for col in self.categorical_features:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return df
    
    def encode_target_variable(self, df, fit=True):
        """
        Encode target variable to binary (0, 1).
        
        Rationale: Binary encoding required for classification algorithms.
        Maps "- 50000." to 0 and "50000+." to 1.
        """
        df = df.copy()
        
        if fit:
            le = LabelEncoder()
            df['label'] = le.fit_transform(df['label'])
            self.label_encoders['label'] = le
            
            # Verify encoding
            print(f"\nTarget encoding:")
            for i, class_name in enumerate(le.classes_):
                print(f"  {class_name} -> {i}")
        else:
            le = self.label_encoders['label']
            df['label'] = le.transform(df['label'])
        
        return df
    
    def scale_numeric_features(self, df, fit=True):
        """
        Standardize numeric features (mean=0, std=1).
        
        Rationale: Different scales (age: 0-90, dividends: 0-99999)
        need normalization for distance-based algorithms and faster
        convergence in gradient-based methods.
        """
        df = df.copy()
        
        if fit:
            df[self.numeric_features] = self.numeric_scaler.fit_transform(
                df[self.numeric_features]
            )
        else:
            df[self.numeric_features] = self.numeric_scaler.transform(
                df[self.numeric_features]
            )
        
        return df
    
    def fit_transform(self, df):
        """
        Fit preprocessing pipeline and transform data.
        
        This method should only be called on training data.
        """
        print("Fitting preprocessing pipeline...")
        print("=" * 50)
        
        # Step 1: Handle duplicates
        print("\n1. Handling duplicates...")
        df = self.handle_duplicates(df)
        
        # Step 2: Identify and handle missing values
        print("\n2. Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Step 3: Categorize features
        print("\n3. Categorizing features...")
        self.create_feature_groups(df)
        
        # Step 4: Impute missing values
        print("\n4. Imputing missing values...")
        df = self.impute_missing_values(df)
        
        # Step 5: Encode target
        print("\n5. Encoding target variable...")
        df = self.encode_target_variable(df, fit=True)
        
        # Step 6: Encode categorical features
        print("\n6. Encoding categorical features...")
        df = self.encode_categorical_variables(df, fit=True)
        
        # Step 7: Scale numeric features
        print("\n7. Scaling numeric features...")
        df = self.scale_numeric_features(df, fit=True)
        
        print("\n" + "=" * 50)
        print("Preprocessing complete!")
        print(f"Final dataset shape: {df.shape}")
        
        return df
    
    def transform(self, df):
        """
        Transform new data using fitted pipeline.
        
        Use this method for validation/test data.
        """
        print("Transforming data using fitted pipeline...")
        
        df = self.handle_missing_values(df)
        df = self.impute_missing_values(df)
        df = self.encode_target_variable(df, fit=False)
        df = self.encode_categorical_variables(df, fit=False)
        df = self.scale_numeric_features(df, fit=False)
        
        print(f"Transformed dataset shape: {df.shape}")
        return df
    
    def save_pipeline(self, filepath):
        """Save fitted pipeline for reproducibility."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"\nPipeline saved to: {filepath}")
    
    @staticmethod
    def load_pipeline(filepath):
        """Load fitted pipeline."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_train_test_split(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    
    Rationale:
    - Stratification maintains class balance in both sets
    - 80/20 split provides sufficient training data while
      reserving adequate test set for reliable evaluation
    - Fixed random seed ensures reproducibility
    """
    # Separate features, target, and weights
    X = df.drop(['label', 'weight'], axis=1)
    y = df['label']
    weights = df['weight']
    
    # Stratified split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"\nTrain/Test split:")
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    print(f"\nClass distribution in training set:")
    print(y_train.value_counts(normalize=True) * 100)
    print(f"\nClass distribution in test set:")
    print(y_test.value_counts(normalize=True) * 100)
    
    return X_train, X_test, y_train, y_test, w_train, w_test


def compute_feature_correlations(df, output_path):
    """
    Compute and save correlation matrix for numeric features.
    
    Rationale: Understanding feature correlations helps identify:
    - Redundant features that can be removed
    - Multicollinearity issues
    - Potential feature interactions
    """
    # Select numeric columns only (exclude target and weight)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['label', 'weight', 'year']]
    
    corr_matrix = df[numeric_cols].corr()
    
    # Save correlation matrix
    corr_matrix.to_csv(output_path)
    print(f"\nCorrelation matrix saved to: {output_path}")
    
    # Identify highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (|r| > 0.7):")
        for pair in high_corr_pairs:
            print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print("\nNo highly correlated features found (|r| > 0.7)")
    
    return corr_matrix


def main():
    """
    Main preprocessing execution.
    """
    print("Census Income Data Preprocessing Pipeline")
    print("=" * 70)
    
    # Paths
    data_path = Path("data/raw/census-bureau.data")
    columns_path = Path("data/raw/census-bureau.columns")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading raw data...")
    with open(columns_path, 'r') as f:
        columns = [line.strip() for line in f.readlines()]
    
    df = pd.read_csv(data_path, names=columns, skipinitialspace=True)
    print(f"Loaded {len(df):,} records with {len(columns)} columns")
    
    # Initialize and fit preprocessor
    preprocessor = CensusDataPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    
    # Compute correlations before train/test split
    print("\n" + "=" * 70)
    print("Computing feature correlations...")
    corr_matrix = compute_feature_correlations(
        df_processed, 
        output_dir / "feature_correlations.csv"
    )
    
    # Create train/test split
    print("\n" + "=" * 70)
    print("Creating train/test split...")
    X_train, X_test, y_train, y_test, w_train, w_test = create_train_test_split(df_processed)
    
    # Save processed datasets
    print("\n" + "=" * 70)
    print("Saving processed datasets...")
    
    train_df = X_train.copy()
    train_df['label'] = y_train.values
    train_df['weight'] = w_train.values
    train_df.to_csv(output_dir / "train_data.csv", index=False)
    print(f"  Saved: {output_dir / 'train_data.csv'}")
    
    test_df = X_test.copy()
    test_df['label'] = y_test.values
    test_df['weight'] = w_test.values
    test_df.to_csv(output_dir / "test_data.csv", index=False)
    print(f"  Saved: {output_dir / 'test_data.csv'}")
    
    # Save preprocessor
    preprocessor.save_pipeline(output_dir / "preprocessor.pkl")
    
    # Save preprocessing metadata
    metadata = {
        "total_records": len(df),
        "processed_records": len(df_processed),
        "records_removed": len(df) - len(df_processed),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "numeric_features": preprocessor.numeric_features,
        "categorical_features": preprocessor.categorical_features,
        "target_classes": preprocessor.label_encoders['label'].classes_.tolist(),
        "class_distribution_train": y_train.value_counts().to_dict(),
        "class_distribution_test": y_test.value_counts().to_dict()
    }
    
    with open(output_dir / "preprocessing_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {output_dir / 'preprocessing_metadata.json'}")
    
    print("\n" + "=" * 70)
    print("Preprocessing pipeline complete!")
    print("\nGenerated files:")
    print(f"  - {output_dir / 'train_data.csv'}")
    print(f"  - {output_dir / 'test_data.csv'}")
    print(f"  - {output_dir / 'preprocessor.pkl'}")
    print(f"  - {output_dir / 'preprocessing_metadata.json'}")
    print(f"  - {output_dir / 'feature_correlations.csv'}")
    print("\nReady for model training!")


if __name__ == "__main__":
    main()