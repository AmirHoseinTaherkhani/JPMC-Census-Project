# Census Income Classification and Segmentation

A machine learning project that predicts income levels and segments customers using U.S. Census data from 1994-1995. Developed as part of the JPMorgan Chase Data Science Challenge.

## Project Overview

This project delivers two complementary approaches to customer targeting:

1. **Classification Model**: An XGBoost classifier that predicts whether individuals earn ≥$50K annually with 93.8% discrimination accuracy (ROC-AUC). The model includes threshold optimization analysis to balance precision and recall for different business scenarios.

2. **Customer Segmentation**: A K-means clustering model that identifies three distinct customer segments based on lifecycle stages, enabling differentiated marketing strategies.

## Key Findings

- **Model Performance**: Achieved ROC-AUC of 0.938 and PR-AUC of 0.630 despite severe class imbalance (15:1 ratio)
- **Threshold Optimization**: Identified that business-optimal thresholds differ significantly from statistical defaults
  - Conservative (0.87): 70.1% precision, 46.4% recall
  - Balanced (0.81): 64.9% precision, 51.8% recall  
  - Aggressive (0.71): 58.0% precision, 58.3% recall
- **Customer Segments**: Three lifecycle-based segments with high-income rates ranging from 0.05% to 13.3%
- **Top Predictors**: Weeks worked per year (16%), occupation (6.7%), and capital gains (4.9%)

## Repository Structure

```
.
├── README.md
├── environment.yml                     # Conda environment specification
├── ML-TakehomeProject.pdf              # Project requirements and dataset description
├── Project Report.pdf                  # Comprehensive analysis report
└── src/                                # Source code
    ├── 01_initial_eda.py               # Data cleaning
    ├── 02_generate_visualization.py    # Data visualization
    ├── 03_preprocessing_pipeline.py    # Preprocessing pipeline
    ├── 04_xgboost_classification.py    # XGB Classification model
    └── 05_customer_segmentation.py     # K-means clustering
```

## Dataset

The analysis uses approximately 200,000 records from the 1994-1995 U.S. Census Bureau Current Population Surveys. Each record contains:

- **40 features** including demographics, education, employment, and financial information
- **Binary target variable**: Income <$50K vs. ≥$50K
- **Population weights**: Accounting for stratified sampling design

**Key Challenges**:
- Severe class imbalance (93.8% low income, 6.2% high income)
- Multiple missing value encodings ("Not in universe", "?", 0)
- Categorical features with high cardinality (17 education levels, 15 occupations)

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AmirHoseinTaherkhani/JPMC-Census-Project
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate census-income
```

3. Verify installation:
```bash
python -c "import xgboost; import sklearn; print('Setup successful!')"
```

## Methodology

### Classification Model

- **Algorithm**: XGBoost with gradient boosting
- **Optimization**: Bayesian hyperparameter search (20 iterations, 3-fold CV)
- **Class Imbalance**: scale_pos_weight=14.85 to upweight minority class
- **Features**: 40 features including label-encoded categoricals
- **Evaluation**: ROC-AUC, PR-AUC, precision, recall, F1-score

### Customer Segmentation

- **Algorithm**: K-means clustering (k=3)
- **Features**: 15 demographic, employment, and financial features
- **Selection Criteria**: Silhouette score, Calinski-Harabasz index, Davies-Bouldin index
- **Segments**: Older adults (24.4%), working professionals (43.3%), early career (32.4%)

## Results Summary

### Classification Performance

| Threshold | Precision | Recall | F1-Score | Use Case |
|-----------|-----------|--------|----------|----------|
| 0.50 (default) | 47.8% | 67.5% | 56.0% | Baseline |
| 0.71 (F1-optimal) | 58.0% | 58.3% | 58.2% | Balanced |
| 0.81 (recommended) | 64.9% | 51.8% | 57.6% | Business-focused |
| 0.87 (conservative) | 70.1% | 46.4% | 55.8% | High precision |

### Customer Segments

| Segment | Size | High Income Rate | Key Characteristics |
|---------|------|------------------|---------------------|
| Segment 0 | 24.4% | 2.1% | Older adults, reduced work, dividend income |
| Segment 1 | 43.3% | 13.3% | Working professionals, full-year employment |
| Segment 2 | 32.4% | 0.05% | Early career, low work engagement |

## Technical Highlights

- **Handles severe class imbalance** through strategic upweighting
- **Preserves population representativeness** using Census sample weights
- **Optimizes business objectives** rather than just statistical metrics
- **Provides interpretable results** through feature importance analysis
- **Offers flexible deployment** with multiple threshold options

## Author

**Amir Taherkhani**

JPMorgan Chase Data Science Challenge - November 2025

---

**Note**: This is a demonstration project using historical Census data. Any deployment would require updated data, legal review, and validation of assumptions.