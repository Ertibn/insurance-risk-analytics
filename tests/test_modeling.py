"""
Unit tests for modeling module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling import (
    InsuranceModelingPipeline,
    engineer_features,
    calculate_risk_based_premium,
)


@pytest.fixture
def sample_modeling_data():
    """Create sample data for modeling tests."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "VehicleAge": np.random.randint(1, 20, n_samples),
        "Premium": np.random.uniform(1000, 5000, n_samples),
        "Claims": np.random.uniform(0, 3000, n_samples),
        "Province": np.random.choice(["GP", "WC", "KZN"], n_samples),
        "Gender": np.random.choice(["M", "F"], n_samples),
    }

    X = pd.DataFrame(data)
    y = X["Claims"]
    X = X.drop(columns=["Claims"])

    return X, y


class TestInsuranceModelingPipeline:
    """Test InsuranceModelingPipeline class."""

    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = InsuranceModelingPipeline(random_state=42)
        assert pipeline.random_state == 42
        assert len(pipeline.models) == 0

    def test_preprocess_features(self, sample_modeling_data):
        """Test feature preprocessing."""
        X, _ = sample_modeling_data
        pipeline = InsuranceModelingPipeline()

        X_processed = pipeline.preprocess_features(X, fit=True)

        assert X_processed.shape == X.shape
        assert pipeline.feature_names is not None
        assert len(pipeline.encoders) > 0

    def test_train_linear_regression(self, sample_modeling_data):
        """Test Linear Regression training."""
        X, y = sample_modeling_data
        pipeline = InsuranceModelingPipeline()

        X_processed = pipeline.preprocess_features(X, fit=True)

        # Split data
        split_idx = int(0.8 * len(X_processed))
        X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        metrics = pipeline.train_linear_regression(X_train, y_train, X_test, y_test)

        assert "model" in metrics
        assert "test_rmse" in metrics
        assert "test_r2" in metrics
        assert metrics["test_rmse"] > 0
        assert -1 <= metrics["test_r2"] <= 1

    def test_train_random_forest_regressor(self, sample_modeling_data):
        """Test Random Forest Regressor training."""
        X, y = sample_modeling_data
        pipeline = InsuranceModelingPipeline()

        X_processed = pipeline.preprocess_features(X, fit=True)

        split_idx = int(0.8 * len(X_processed))
        X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        metrics = pipeline.train_random_forest_regressor(
            X_train, y_train, X_test, y_test, n_estimators=10
        )

        assert "model" in metrics
        assert "test_rmse" in metrics
        assert "feature_importance" in metrics
        assert metrics["test_rmse"] > 0

    def test_train_random_forest_classifier(self, sample_modeling_data):
        """Test Random Forest Classifier training."""
        X, y = sample_modeling_data
        pipeline = InsuranceModelingPipeline()

        X_processed = pipeline.preprocess_features(X, fit=True)

        # Create binary target
        y_binary = (y > y.median()).astype(int)

        split_idx = int(0.8 * len(X_processed))
        X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_test = y_binary[:split_idx], y_binary[split_idx:]

        metrics = pipeline.train_random_forest_classifier(
            X_train, y_train, X_test, y_test, n_estimators=10
        )

        assert "model" in metrics
        assert "test_accuracy" in metrics
        assert "test_f1" in metrics
        assert 0 <= metrics["test_accuracy"] <= 1

    def test_get_model_comparison(self, sample_modeling_data):
        """Test model comparison."""
        X, y = sample_modeling_data
        pipeline = InsuranceModelingPipeline()

        X_processed = pipeline.preprocess_features(X, fit=True)

        split_idx = int(0.8 * len(X_processed))
        X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        metrics_lr = pipeline.train_linear_regression(X_train, y_train, X_test, y_test)
        metrics_rf = pipeline.train_random_forest_regressor(
            X_train, y_train, X_test, y_test, n_estimators=10
        )

        comparison = pipeline.get_model_comparison([metrics_lr, metrics_rf])

        assert len(comparison) == 2
        assert "Model" in comparison.columns
        assert "Test RMSE" in comparison.columns


class TestEngineerFeatures:
    """Test feature engineering functions."""

    def test_engineer_features(self):
        """Test feature engineering."""
        data = {
            "RegistrationYear": [2010, 2012, 2015],
            "TransactionMonth": [1, 6, 12],
        }
        df = pd.DataFrame(data)

        df_engineered = engineer_features(df)

        assert "VehicleAge" in df_engineered.columns
        assert "PolicyDuration" in df_engineered.columns
        assert df_engineered.loc[0, "VehicleAge"] == 5


class TestRiskBasedPremium:
    """Test risk-based premium calculation."""

    def test_calculate_risk_based_premium(self):
        """Test premium calculation."""
        claim_prob = 0.1
        severity = 5000

        premium = calculate_risk_based_premium(claim_prob, severity)

        # Pure premium = 0.1 * 5000 = 500
        # Loading = 500 * (0.15 + 0.20) = 175
        # Total = 675
        expected = 500 + 175
        assert premium == pytest.approx(expected)

    def test_calculate_risk_based_premium_custom_loading(self):
        """Test premium calculation with custom loading."""
        claim_prob = 0.2
        severity = 10000
        expense_loading = 0.10
        profit_margin = 0.15

        premium = calculate_risk_based_premium(
            claim_prob, severity, expense_loading, profit_margin
        )

        # Pure premium = 0.2 * 10000 = 2000
        # Loading = 2000 * (0.10 + 0.15) = 500
        # Total = 2500
        expected = 2000 + 500
        assert premium == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
