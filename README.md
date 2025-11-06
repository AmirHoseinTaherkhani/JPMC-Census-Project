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
├── environment.yml              # Conda environment specification
├── ML-TakehomeProject.pdf       # Project requirements and dataset description
├── Project Report.pdf           # Comprehensive analysis report
└── src/                         # Source code
    ├── data_preprocessing.py    # Data cleaning and feature engineering
    ├── model_training.py        # XGBoost model development
    ├── threshold_optimization.py # Threshold analysis
    ├── customer_segmentation.py # K-means clustering
    ├── visualization.py         # Plotting and charts
    └── evaluation.py            # Model evaluation metrics
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
git clone https://github.com/yourusername/census-income-classification.git
cd census-income-classification
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

## Usage

### 1. Data Preprocessing

```python
from src.data_preprocessing import preprocess_data

# Load and clean data
X_train, X_test, y_train, y_test, sample_weights = preprocess_data('data/census_data.csv')
```

### 2. Train Classification Model

```python
from src.model_training import train_xgboost_model

# Train model with optimized hyperparameters
model = train_xgboost_model(X_train, y_train, sample_weights)
```

### 3. Optimize Decision Threshold

```python
from src.threshold_optimization import find_optimal_threshold

# Analyze precision-recall tradeoffs
optimal_threshold = find_optimal_threshold(model, X_test, y_test)
```

### 4. Customer Segmentation

```python
from src.customer_segmentation import create_segments

# Generate three customer segments
segments = create_segments(X_train, n_clusters=3)
```

### 5. Evaluate and Visualize

```python
from src.evaluation import evaluate_model
from src.visualization import plot_performance

# Evaluate model performance
metrics = evaluate_model(model, X_test, y_test, threshold=0.81)

# Generate visualizations
plot_performance(model, X_test, y_test)
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

## Limitations

- Data from 1994-1995 may not reflect current economic conditions
- Model includes protected characteristics (sex, race) requiring legal review
- Assumes relatively stable relationships between features and income
- Performance depends on similarity between training and deployment populations

## Future Work

- Update analysis with current Census data
- Implement fairness constraints to address protected characteristics
- Develop ensemble methods combining multiple algorithms
- Create real-time prediction API
- Build interactive dashboard for threshold selection
- Validate segments through A/B testing

## Author

**Amir Taherkhani**

JPMorgan Chase Data Science Challenge - November 2025

## License

This project is available for educational and portfolio purposes.

## Acknowledgments

- U.S. Census Bureau for the dataset
- JPMorgan Chase for the challenge opportunity
- XGBoost and scikit-learn communities for excellent documentation

---

**Note**: This is a demonstration project using historical Census data. Any deployment would require updated data, legal review, and validation of assumptions.