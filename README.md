# Insurance Risk Analytics & Predictive Modeling

## Project Overview

This project is part of the **10 Academy: Artificial Intelligence Mastery - Week 3 Challenge**. We analyze 18 months of historical insurance claim data (Feb 2014 – Aug 2015) from AlphaCare Insurance Solutions (ACIS) to:

1. **Optimize marketing strategy** by identifying low-risk customer segments
2. **Develop risk-based pricing models** that predict claim severity and frequency
3. **Provide data-driven recommendations** for premium optimization and product refinement

## Business Context

ACIS is preparing for aggressive growth in the South African auto-insurance market. This analysis provides evidence-driven strategies to:
- Move beyond intuition-based pricing toward analytics-driven decisions
- Identify high-value, low-risk customer segments for targeted marketing
- Build predictive models for dynamic, risk-based premium calculation

## Key Metrics

- **Loss Ratio** = TotalClaims / TotalPremium (portfolio profitability)
- **Margin** = TotalPremium − TotalClaims (per-policy profit)
- **Claim Frequency** = proportion of policies with at least one claim
- **Claim Severity** = average claim amount, given a claim occurred

## Project Structure

```
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── data/                          # Tracked by DVC, not Git
│   └── insurance_data.csv
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb # A/B Hypothesis Testing
│   └── 03_modeling.ipynb         # Statistical Modeling & Risk-Based Pricing
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading utilities
│   ├── eda_utils.py              # EDA helper functions
│   ├── hypothesis_tests.py       # Statistical testing functions
│   └── modeling.py               # Modeling utilities
├── reports/
│   └── final_report.md           # Final business-facing report
├── tests/
│   ├── test_data_loader.py
│   └── test_modeling.py
├── .dvc/                         # DVC configuration
├── dvc.yaml                      # DVC pipeline definition
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd insurance-risk-analytics
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize DVC
```bash
dvc init
dvc remote add -d localstorage /path/to/local/storage
```

### 5. Pull Data
```bash
dvc pull
```

## Data Pipeline & Reproducibility

This project uses **Data Version Control (DVC)** to ensure reproducibility and auditability—critical requirements in regulated industries like insurance.

### DVC Workflow

1. **Track raw data**: `dvc add data/insurance_data.csv`
2. **Create versions**: Raw → Cleaned → Processed
3. **Push to remote**: `dvc push`
4. **Reproduce pipeline**: `dvc repro`

### Reproducing the Analysis

```bash
# Pull the exact data version used in this analysis
dvc pull

# Run the full pipeline
dvc repro

# Or run individual notebooks
jupyter notebook notebooks/01_eda.ipynb
```

## Tasks & Deliverables

### Task 1: Git, GitHub & Exploratory Data Analysis
- ✅ GitHub repository with CI/CD pipeline
- ✅ EDA covering data quality, distributions, relationships, and geographic trends
- ✅ 3+ insight-driven visualizations
- **Branch**: `task-1`

### Task 2: Data Version Control (DVC)
- ✅ DVC initialization and remote storage setup
- ✅ Data versioning with `.dvc` files
- ✅ Reproducible data pipeline documentation
- **Branch**: `task-2`

### Task 3: A/B Hypothesis Testing
- Statistical validation of risk drivers across provinces, zip codes, and gender
- Chi-squared, t-tests, and z-tests with p-value thresholds (α = 0.05)
- Business-facing interpretations for rejected hypotheses
- **Branch**: `task-3`

### Task 4: Statistical Modeling & Risk-Based Pricing
- Claim Severity Prediction (Linear Regression, Random Forest, XGBoost)
- Claim Probability Classification
- SHAP/LIME feature importance analysis
- **Branch**: `task-4`

## Key Findings (EDA)

### Loss Ratio by Province
- Highest risk: [Province with highest loss ratio]
- Lowest risk: [Province with lowest loss ratio]

### Claim Severity Distribution
- Mean claim amount: R[amount]
- Median claim amount: R[amount]
- Notable outliers detected in [vehicle type/region]

### Geographic Trends
- [Key geographic insight]
- [Regional risk variation]

## Hypothesis Testing Results

| Hypothesis | Test | p-value | Decision | Insight |
|-----------|------|---------|----------|---------|
| Risk differences across provinces | Chi-squared | [p] | [Reject/Fail to Reject] | [Business insight] |
| Risk differences by zip code | Chi-squared | [p] | [Reject/Fail to Reject] | [Business insight] |
| Margin differences by zip code | t-test | [p] | [Reject/Fail to Reject] | [Business insight] |
| Risk differences by gender | Chi-squared | [p] | [Reject/Fail to Reject] | [Business insight] |

## Modeling Results

### Model Comparison

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| Linear Regression | [value] | [value] | [value] |
| Random Forest | [value] | [value] | [value] |
| XGBoost | [value] | [value] | [value] |

### Top 5 Features (SHAP Analysis)
1. [Feature] - [Impact description]
2. [Feature] - [Impact description]
3. [Feature] - [Impact description]
4. [Feature] - [Impact description]
5. [Feature] - [Impact description]

## Business Recommendations

1. **Segmentation Strategy**: Target low-risk provinces/zip codes with reduced premiums
2. **Pricing Adjustment**: Implement risk-based premium calculation using predicted severity
3. **Product Refinement**: Adjust coverage limits based on vehicle type and region
4. **Marketing Focus**: Concentrate acquisition efforts on high-margin segments

## Limitations & Future Work

### Current Limitations
- Analysis limited to 18-month historical period (Feb 2014 – Aug 2015)
- External factors (economic conditions, policy changes) not captured
- Model performance may degrade with market shifts

### Future Enhancements
- Incorporate external economic indicators
- Develop time-series models for seasonal trends
- Implement real-time model monitoring and retraining
- Expand to multi-line insurance products

## CI/CD Pipeline

GitHub Actions automatically runs on every push:
- **Linting**: Black, Flake8, Pylint
- **Testing**: Pytest with coverage reporting
- **Status**: Check `.github/workflows/ci.yml` for details

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit with descriptive messages: `git commit -m "feat: add new analysis"`
3. Push and create a Pull Request
4. Ensure CI/CD pipeline passes before merging

## Team

- **Tutors**: Kerod, Mahbubah, Feven
- **Challenge Period**: 20 May – 26 May 2026
- **Interim Submission**: 25 May 2026, 8:00 PM UTC
- **Final Submission**: 26 May 2026, 8:00 PM UTC

## References

- [Insurance Analytics](https://example.com)
- [DVC Official Documentation](https://dvc.org)
- [SHAP Documentation](https://shap.readthedocs.io)
- [A/B Testing Best Practices](https://example.com)

## License

This project is part of the 10 Academy curriculum. All rights reserved.

---

**Last Updated**: May 26, 2026
**Status**: In Progress (Task 1-2 Complete, Task 3-4 In Progress)
