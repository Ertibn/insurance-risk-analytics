"""
Execute Tasks 3 & 4: Hypothesis Testing and Modeling
This script runs the complete analysis pipeline and generates results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.hypothesis_tests import (
    test_risk_by_province,
    test_risk_by_zipcode,
    test_margin_by_zipcode,
    test_risk_by_gender,
)
from src.modeling import engineer_features, calculate_risk_based_premium

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def execute_task3_hypothesis_testing():
    """Execute Task 3: Hypothesis Testing"""
    print("\n" + "="*70)
    print("TASK 3: A/B HYPOTHESIS TESTING")
    print("="*70)
    
    # Load data
    data_path = 'data/processed/cleaned_data.csv'
    df = pd.read_csv(data_path)
    print(f"\nData loaded: {df.shape}")
    
    # Initialize results
    results = {
        "hypothesis_tests": [],
        "summary": {}
    }
    
    # Test 1: Risk differences across provinces
    print("\n[1/4] Testing risk differences across provinces...")
    try:
        test1_result = test_risk_by_province(df)
        results["hypothesis_tests"].append({
            "hypothesis": "Risk differences across provinces",
            "test": "Chi-squared",
            "p_value": float(test1_result.get("p_value", 0)),
            "decision": "Reject H₀" if test1_result.get("p_value", 1) < 0.05 else "Fail to Reject H₀",
            "insight": test1_result.get("insight", "")
        })
        print(f"   ✓ p-value: {test1_result.get('p_value', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Test 2: Risk differences by zip code
    print("\n[2/4] Testing risk differences by zip code...")
    try:
        test2_result = test_risk_by_zipcode(df)
        results["hypothesis_tests"].append({
            "hypothesis": "Risk differences by zip code",
            "test": "Chi-squared",
            "p_value": float(test2_result.get("p_value", 0)),
            "decision": "Reject H₀" if test2_result.get("p_value", 1) < 0.05 else "Fail to Reject H₀",
            "insight": test2_result.get("insight", "")
        })
        print(f"   ✓ p-value: {test2_result.get('p_value', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Test 3: Margin differences by zip code
    print("\n[3/4] Testing margin differences by zip code...")
    try:
        test3_result = test_margin_by_zipcode(df)
        results["hypothesis_tests"].append({
            "hypothesis": "Margin differences by zip code",
            "test": "t-test",
            "p_value": float(test3_result.get("p_value", 0)),
            "decision": "Reject H₀" if test3_result.get("p_value", 1) < 0.05 else "Fail to Reject H₀",
            "insight": test3_result.get("insight", "")
        })
        print(f"   ✓ p-value: {test3_result.get('p_value', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Test 4: Risk differences by gender
    print("\n[4/4] Testing risk differences by gender...")
    try:
        test4_result = test_risk_by_gender(df)
        results["hypothesis_tests"].append({
            "hypothesis": "Risk differences by gender",
            "test": "Chi-squared",
            "p_value": float(test4_result.get("p_value", 0)),
            "decision": "Reject H₀" if test4_result.get("p_value", 1) < 0.05 else "Fail to Reject H₀",
            "insight": test4_result.get("insight", "")
        })
        print(f"   ✓ p-value: {test4_result.get('p_value', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Save results
    results_path = 'data/processed/hypothesis_testing_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Hypothesis testing results saved to {results_path}")
    
    return results, df

def execute_task4_modeling(df):
    """Execute Task 4: Statistical Modeling & Risk-Based Pricing"""
    print("\n" + "="*70)
    print("TASK 4: STATISTICAL MODELING & RISK-BASED PRICING")
    print("="*70)
    
    # Engineer features
    print("\n[1/5] Engineering features...")
    df_features = engineer_features(df)
    print(f"   ✓ Features engineered: {df_features.shape[1]} columns")
    
    # Prepare data for modeling
    print("\n[2/5] Preparing data for modeling...")
    
    # Select features for modeling
    feature_cols = [col for col in df_features.columns 
                   if col not in ['CustomerID', 'TotalClaims', 'ClaimIndicator', 
                                 'TransactionDate', 'Claimed']]
    
    # Handle categorical variables
    df_model = df_features.copy()
    categorical_cols = df_model[feature_cols].select_dtypes(include=['object']).columns
    df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    
    # Update feature columns after encoding
    feature_cols = [col for col in df_model.columns 
                   if col not in ['CustomerID', 'TotalClaims', 'ClaimIndicator', 
                                 'TransactionDate', 'Claimed']]
    
    print(f"   ✓ Data prepared: {len(feature_cols)} features")
    
    # Task 4a: Claim Severity Prediction
    print("\n[3/5] Building Claim Severity Prediction Model...")
    
    # Filter for policies with claims
    df_claims = df_model[df_model['ClaimIndicator'] == 1].copy()
    
    if len(df_claims) > 0:
        X_severity = df_claims[feature_cols]
        y_severity = df_claims['TotalClaims']
        
        # Split data
        X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
            X_severity, y_severity, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_sev_scaled = scaler.fit_transform(X_train_sev)
        X_test_sev_scaled = scaler.transform(X_test_sev)
        
        # Train Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_sev_scaled, y_train_sev)
        y_pred_lr = lr_model.predict(X_test_sev_scaled)
        
        lr_rmse = np.sqrt(mean_squared_error(y_test_sev, y_pred_lr))
        lr_r2 = r2_score(y_test_sev, y_pred_lr)
        lr_mae = mean_absolute_error(y_test_sev, y_pred_lr)
        
        print(f"   ✓ Linear Regression - RMSE: {lr_rmse:.2f}, R²: {lr_r2:.4f}, MAE: {lr_mae:.2f}")
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_sev, y_train_sev)
        y_pred_rf = rf_model.predict(X_test_sev)
        
        rf_rmse = np.sqrt(mean_squared_error(y_test_sev, y_pred_rf))
        rf_r2 = r2_score(y_test_sev, y_pred_rf)
        rf_mae = mean_absolute_error(y_test_sev, y_pred_rf)
        
        print(f"   ✓ Random Forest - RMSE: {rf_rmse:.2f}, R²: {rf_r2:.4f}, MAE: {rf_mae:.2f}")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        print(f"   ✓ Top 5 Features: {', '.join(feature_importance.head(5)['feature'].tolist())}")
    else:
        print("   ✗ No claims data available for severity modeling")
        lr_rmse = lr_r2 = lr_mae = rf_rmse = rf_r2 = rf_mae = 0
        feature_importance = pd.DataFrame()
    
    # Task 4b: Claim Probability Prediction
    print("\n[4/5] Building Claim Probability Prediction Model...")
    
    X_prob = df_model[feature_cols]
    y_prob = df_model['ClaimIndicator']
    
    X_train_prob, X_test_prob, y_train_prob, y_test_prob = train_test_split(
        X_prob, y_prob, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_prob = StandardScaler()
    X_train_prob_scaled = scaler_prob.fit_transform(X_train_prob)
    X_test_prob_scaled = scaler_prob.transform(X_test_prob)
    
    # Train logistic regression for probability
    from sklearn.linear_model import LogisticRegression
    lr_prob = LogisticRegression(random_state=42, max_iter=1000)
    lr_prob.fit(X_train_prob_scaled, y_train_prob)
    
    prob_accuracy = lr_prob.score(X_test_prob_scaled, y_test_prob)
    print(f"   ✓ Claim Probability Model - Accuracy: {prob_accuracy:.4f}")
    
    # Task 4c: Risk-Based Premium Calculation
    print("\n[5/5] Calculating Risk-Based Premiums...")
    
    # Calculate risk-based premiums
    df_model['PredictedClaimProbability'] = lr_prob.predict_proba(
        scaler_prob.transform(X_prob)
    )[:, 1]
    
    if len(df_claims) > 0:
        df_model['PredictedSeverity'] = 0.0
        df_model.loc[df_model['ClaimIndicator'] == 1, 'PredictedSeverity'] = rf_model.predict(
            X_prob[df_model['ClaimIndicator'] == 1]
        )
    
    df_model['RiskBasedPremium'] = calculate_risk_based_premium(
        df_model['PredictedClaimProbability'],
        df_model.get('PredictedSeverity', 0),
        expense_loading=0.15,
        profit_margin=0.20
    )
    
    print(f"   ✓ Risk-based premiums calculated")
    print(f"   ✓ Mean Risk-Based Premium: R{df_model['RiskBasedPremium'].mean():.2f}")
    print(f"   ✓ Median Risk-Based Premium: R{df_model['RiskBasedPremium'].median():.2f}")
    
    # Save modeling results
    modeling_results = {
        "claim_severity_model": {
            "linear_regression": {
                "rmse": float(lr_rmse),
                "r2": float(lr_r2),
                "mae": float(lr_mae)
            },
            "random_forest": {
                "rmse": float(rf_rmse),
                "r2": float(rf_r2),
                "mae": float(rf_mae)
            }
        },
        "claim_probability_model": {
            "accuracy": float(prob_accuracy)
        },
        "top_features": feature_importance.to_dict('records') if len(feature_importance) > 0 else [],
        "risk_based_premium_stats": {
            "mean": float(df_model['RiskBasedPremium'].mean()),
            "median": float(df_model['RiskBasedPremium'].median()),
            "std": float(df_model['RiskBasedPremium'].std()),
            "min": float(df_model['RiskBasedPremium'].min()),
            "max": float(df_model['RiskBasedPremium'].max())
        }
    }
    
    results_path = 'data/processed/modeling_results.json'
    with open(results_path, 'w') as f:
        json.dump(modeling_results, f, indent=2)
    print(f"\n✓ Modeling results saved to {results_path}")
    
    return modeling_results

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("INSURANCE RISK ANALYTICS - TASKS 3 & 4 EXECUTION")
    print("="*70)
    
    # Execute Task 3
    hypothesis_results, df = execute_task3_hypothesis_testing()
    
    # Execute Task 4
    modeling_results = execute_task4_modeling(df)
    
    # Print summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    print("\n✓ Task 3: Hypothesis Testing - COMPLETED")
    print(f"  - Tests performed: {len(hypothesis_results['hypothesis_tests'])}")
    for test in hypothesis_results['hypothesis_tests']:
        print(f"    • {test['hypothesis']}: {test['decision']} (p={test['p_value']:.4f})")
    
    print("\n✓ Task 4: Statistical Modeling - COMPLETED")
    print(f"  - Claim Severity Model (Random Forest):")
    print(f"    • RMSE: {modeling_results['claim_severity_model']['random_forest']['rmse']:.2f}")
    print(f"    • R²: {modeling_results['claim_severity_model']['random_forest']['r2']:.4f}")
    print(f"  - Claim Probability Model:")
    print(f"    • Accuracy: {modeling_results['claim_probability_model']['accuracy']:.4f}")
    print(f"  - Risk-Based Premium:")
    print(f"    • Mean: R{modeling_results['risk_based_premium_stats']['mean']:.2f}")
    print(f"    • Median: R{modeling_results['risk_based_premium_stats']['median']:.2f}")
    
    print("\n" + "="*70)
    print("All tasks completed successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
