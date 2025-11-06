"""
Initial Exploratory Data Analysis for Census Income Dataset
Generates JSON summary for preprocessing pipeline development

Author: Senior Data Scientist, JP Morgan
Date: November 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def analyze_column_type(series):
    """
    Determine the practical data type and characteristics of a column.
    
    Rationale: Census data often has numeric codes representing categories,
    so we need to distinguish between true numeric features and encoded categoricals.
    """
    # Basic type info
    dtype_str = str(series.dtype)
    
    # Check for numeric types
    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique()
        total_count = len(series)
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        
        # If few unique values relative to total, likely categorical
        if unique_ratio < 0.05 and unique_count < 50:
            inferred_type = "categorical_numeric"
        else:
            inferred_type = "continuous_numeric"
    else:
        inferred_type = "categorical_string"
    
    return {
        "dtype": dtype_str,
        "inferred_type": inferred_type
    }


def get_missing_info(series):
    """
    Analyze missing value patterns.
    
    Rationale: Census data may use special codes (99, -1, ' ?') for missing values
    rather than explicit NaNs. Need to detect these patterns.
    """
    missing_count = series.isna().sum()
    missing_pct = (missing_count / len(series)) * 100
    
    # Check for common missing value indicators
    potential_missing_indicators = []
    
    if pd.api.types.is_numeric_dtype(series):
        if (series == -1).sum() > 0:
            potential_missing_indicators.append(-1)
        if (series == 99).sum() > 0:
            potential_missing_indicators.append(99)
        if (series == 999).sum() > 0:
            potential_missing_indicators.append(999)
    else:
        # Check for string patterns indicating missing
        value_counts = series.value_counts()
        for val in value_counts.index[:10]:  # Check top 10 values
            if isinstance(val, str) and val.strip() in ['?', ' ?', 'Unknown', 'Not in universe']:
                potential_missing_indicators.append(val)
    
    return {
        "missing_count": int(missing_count),
        "missing_percentage": round(missing_pct, 2),
        "potential_missing_indicators": potential_missing_indicators
    }


def get_basic_stats(series):
    """
    Generate appropriate statistics based on column type.
    """
    stats = {}
    
    if pd.api.types.is_numeric_dtype(series):
        non_missing = series.dropna()
        if len(non_missing) > 0:
            stats = {
                "min": float(non_missing.min()),
                "max": float(non_missing.max()),
                "mean": round(float(non_missing.mean()), 2),
                "median": float(non_missing.median()),
                "std": round(float(non_missing.std()), 2),
                "q25": float(non_missing.quantile(0.25)),
                "q75": float(non_missing.quantile(0.75))
            }
    
    return stats


def get_value_distribution(series, top_n=10):
    """
    Get distribution of values for categorical or low-cardinality numeric columns.
    
    Rationale: Understanding value distributions helps identify:
    - Class imbalances
    - Rare categories that may need grouping
    - Unexpected values or data quality issues
    """
    value_counts = series.value_counts()
    
    if len(value_counts) <= top_n:
        distribution = {str(k): int(v) for k, v in value_counts.items()}
    else:
        top_values = {str(k): int(v) for k, v in value_counts.head(top_n).items()}
        other_count = int(value_counts.iloc[top_n:].sum())
        distribution = {**top_values, "other": other_count}
    
    return {
        "unique_count": int(len(value_counts)),
        "top_values": distribution
    }


def analyze_target_variable(df, target_col):
    """
    Specific analysis for the income target variable.
    
    Rationale: Understanding target distribution is critical for:
    - Identifying class imbalance
    - Planning stratification strategy
    - Setting baseline model performance expectations
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found"}
    
    value_counts = df[target_col].value_counts()
    total = len(df)
    
    distribution = {}
    for label, count in value_counts.items():
        distribution[str(label)] = {
            "count": int(count),
            "percentage": round((count / total) * 100, 2)
        }
    
    return {
        "distribution": distribution,
        "imbalance_ratio": round(float(value_counts.max() / value_counts.min()), 2)
    }


def analyze_weights(df, weight_col):
    """
    Analyze the sample weight column.
    
    Rationale: Weights represent population distribution from stratified sampling.
    Understanding weight distribution is essential for proper model training and evaluation.
    """
    if weight_col not in df.columns:
        return {"error": f"Weight column '{weight_col}' not found"}
    
    weights = df[weight_col].dropna()
    
    return {
        "min": float(weights.min()),
        "max": float(weights.max()),
        "mean": round(float(weights.mean()), 2),
        "median": float(weights.median()),
        "std": round(float(weights.std()), 2),
        "zero_weights": int((weights == 0).sum()),
        "negative_weights": int((weights < 0).sum())
    }


def main():
    """
    Main EDA execution.
    """
    # File paths
    data_path = Path("data/raw/census-bureau.data")
    columns_path = Path("data/raw/census-bureau.columns")
    output_path = Path("outputs/eda_summary.json")
    
    print("Starting EDA for Census Income Dataset...")
    print(f"Loading data from {data_path}")
    
    # Load column names
    with open(columns_path, 'r') as f:
        columns = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(columns)} columns")
    
    # Load data
    df = pd.read_csv(data_path, names=columns, skipinitialspace=True)
    
    print(f"Loaded {len(df)} records")
    print(f"Dataset shape: {df.shape}")
    
    # Initialize EDA summary
    eda_summary = {
        "dataset_info": {
            "total_records": len(df),
            "total_features": len(columns),
            "column_names": columns,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        },
        "columns_analysis": {},
        "data_quality": {
            "total_missing_cells": int(df.isna().sum().sum()),
            "columns_with_missing": [],
            "duplicate_rows": int(df.duplicated().sum())
        }
    }
    
    # Identify target and weight columns based on typical census data structure
    # Last column is usually the target, and there's typically a weight column
    potential_target = columns[-1]
    potential_weight = [col for col in columns if 'weight' in col.lower() or 'wgt' in col.lower()]
    
    print(f"\nIdentified potential target variable: {potential_target}")
    if potential_weight:
        print(f"Identified potential weight variable: {potential_weight[0]}")
    
    # Analyze each column
    print("\nAnalyzing individual columns...")
    for col in columns:
        print(f"  Processing: {col}")
        
        col_analysis = {
            "type_info": analyze_column_type(df[col]),
            "missing_info": get_missing_info(df[col]),
            "basic_stats": get_basic_stats(df[col])
        }
        
        # Add value distribution for categorical or low-cardinality columns
        if col_analysis["type_info"]["inferred_type"] in ["categorical_numeric", "categorical_string"] or \
           df[col].nunique() < 50:
            col_analysis["value_distribution"] = get_value_distribution(df[col])
        
        eda_summary["columns_analysis"][col] = col_analysis
        
        # Track columns with missing values
        if col_analysis["missing_info"]["missing_count"] > 0:
            eda_summary["data_quality"]["columns_with_missing"].append({
                "column": col,
                "missing_count": col_analysis["missing_info"]["missing_count"],
                "missing_percentage": col_analysis["missing_info"]["missing_percentage"]
            })
    
    # Target variable analysis
    print("\nAnalyzing target variable...")
    eda_summary["target_analysis"] = analyze_target_variable(df, potential_target)
    
    # Weight variable analysis
    if potential_weight:
        print("Analyzing weight variable...")
        eda_summary["weight_analysis"] = analyze_weights(df, potential_weight[0])
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(eda_summary, f, indent=2)
    
    print(f"\nEDA summary saved to {output_path}")
    print("\nKey Findings:")
    print(f"  - Total records: {eda_summary['dataset_info']['total_records']:,}")
    print(f"  - Total features: {eda_summary['dataset_info']['total_features']}")
    print(f"  - Columns with missing data: {len(eda_summary['data_quality']['columns_with_missing'])}")
    print(f"  - Duplicate rows: {eda_summary['data_quality']['duplicate_rows']:,}")
    
    if 'distribution' in eda_summary['target_analysis']:
        print(f"\nTarget Variable Distribution:")
        for label, info in eda_summary['target_analysis']['distribution'].items():
            print(f"  - {label}: {info['count']:,} ({info['percentage']}%)")
    
    print("\nEDA complete. Review eda_summary.json for detailed findings.")


if __name__ == "__main__":
    main()