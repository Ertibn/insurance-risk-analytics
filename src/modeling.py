"""
Predictive Modeling Utilities

This module provides functions for building, training, and evaluating
predictive models for claim severity and probability prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings("ignore")


class InsuranceModelingPipeline:
    """Pipeline for building and evaluating insurance predictive models."""

    def __init__(self, random_state: int = 42):
        """
        Initialize modeling pipeline.

        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None

    def preprocess_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Preprocess features: encode categorical, scale numerical.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable (not used, for API consistency).
        fit : bool, default=True
            Whether to fit encoders/scalers.

        Returns
        -------
        pd.DataFrame
            Preprocessed features.
        """
        X_processed = X.copy()

        # Encode categorical variables
        categorical_cols = X_processed.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                X_processed[col] = self.encoders[col].fit_transform(X_processed[col].astype(str))
            else:
                X_processed[col] = self.encoders[col].transform(X_processed[col].astype(str))

        # Scale numerical variables
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
        if fit:
            self.scalers["numerical"] = StandardScaler()
            X_processed[numerical_cols] = self.scalers["numerical"].fit_transform(X_processed[numerical_cols])
        else:
            X_processed[numerical_cols] = self.scalers["numerical"].transform(X_processed[numerical_cols])

        self.feature_names = X_processed.columns.tolist()
        return X_processed

    def train_linear_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """
        Train and evaluate Linear Regression model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target.

        Returns
        -------
        Dict
            Model and evaluation metrics.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "model": model,
            "model_name": "Linear Regression",
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "coefficients": pd.DataFrame({
                "Feature": self.feature_names,
                "Coefficient": model.coef_,
            }).sort_values("Coefficient", key=abs, ascending=False),
        }

        self.models["linear_regression"] = model
        return metrics

    def train_random_forest_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
    ) -> Dict:
        """
        Train and evaluate Random Forest Regressor.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target.
        n_estimators : int, default=100
            Number of trees.
        max_depth : int, optional
            Maximum tree depth.

        Returns
        -------
        Dict
            Model and evaluation metrics.
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "model": model,
            "model_name": "Random Forest Regressor",
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "feature_importance": pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False),
        }

        self.models["random_forest"] = model
        return metrics

    def train_xgboost_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ) -> Dict:
        """
        Train and evaluate XGBoost Regressor.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target.
        n_estimators : int, default=100
            Number of boosting rounds.
        max_depth : int, default=5
            Maximum tree depth.
        learning_rate : float, default=0.1
            Learning rate.

        Returns
        -------
        Dict
            Model and evaluation metrics.
        """
        try:
            import xgboost as xgb

            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            metrics = {
                "model": model,
                "model_name": "XGBoost Regressor",
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
                "feature_importance": pd.DataFrame({
                    "Feature": self.feature_names,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=False),
            }

            self.models["xgboost"] = model
            return metrics

        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    def train_random_forest_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
    ) -> Dict:
        """
        Train and evaluate Random Forest Classifier for claim probability.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target (binary).
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target (binary).
        n_estimators : int, default=100
            Number of trees.
        max_depth : int, optional
            Maximum tree depth.

        Returns
        -------
        Dict
            Model and evaluation metrics.
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": model,
            "model_name": "Random Forest Classifier",
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "train_precision": precision_score(y_train, y_pred_train, zero_division=0),
            "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
            "train_recall": recall_score(y_train, y_pred_train, zero_division=0),
            "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
            "train_f1": f1_score(y_train, y_pred_train, zero_division=0),
            "test_f1": f1_score(y_test, y_pred_test, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_pred_proba_test),
            "feature_importance": pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False),
        }

        self.models["rf_classifier"] = model
        return metrics

    def get_model_comparison(self, metrics_list: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple models.

        Parameters
        ----------
        metrics_list : List[Dict]
            List of metrics dictionaries from training functions.

        Returns
        -------
        pd.DataFrame
            Model comparison table.
        """
        comparison_data = []
        for metrics in metrics_list:
            row = {
                "Model": metrics["model_name"],
                "Train RMSE": metrics.get("train_rmse", np.nan),
                "Test RMSE": metrics.get("test_rmse", np.nan),
                "Train R²": metrics.get("train_r2", np.nan),
                "Test R²": metrics.get("test_r2", np.nan),
                "Train MAE": metrics.get("train_mae", np.nan),
                "Test MAE": metrics.get("test_mae", np.nan),
            }
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_shap_values(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for model interpretability.

        Parameters
        ----------
        model_name : str
            Name of the model ("random_forest" or "xgboost").
        X : pd.DataFrame
            Input features.

        Returns
        -------
        np.ndarray
            SHAP values.
        """
        try:
            import shap

            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found in trained models.")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            return shap_values

        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Returns
    -------
    pd.DataFrame
        Data with engineered features.
    """
    df_engineered = df.copy()

    # Vehicle age
    if "RegistrationYear" in df_engineered.columns:
        df_engineered["VehicleAge"] = 2015 - df_engineered["RegistrationYear"]

    # Policy duration (if TransactionMonth available)
    if "TransactionMonth" in df_engineered.columns:
        df_engineered["PolicyDuration"] = df_engineered["TransactionMonth"]

    # Premium per vehicle
    if "SumInsured" in df_engineered.columns and "NumberOfVehiclesInFleet" in df_engineered.columns:
        df_engineered["PremiumPerVehicle"] = (
            df_engineered["SumInsured"] / (df_engineered["NumberOfVehiclesInFleet"] + 1)
        )

    return df_engineered


def calculate_risk_based_premium(
    claim_probability: float,
    predicted_severity: float,
    expense_loading: float = 0.15,
    profit_margin: float = 0.20,
) -> float:
    """
    Calculate risk-based premium.

    Parameters
    ----------
    claim_probability : float
        Predicted probability of claim (0-1).
    predicted_severity : float
        Predicted claim amount if claim occurs.
    expense_loading : float, default=0.15
        Expense loading factor (15% of pure premium).
    profit_margin : float, default=0.20
        Desired profit margin (20% of pure premium).

    Returns
    -------
    float
        Calculated premium.
    """
    pure_premium = claim_probability * predicted_severity
    total_loading = pure_premium * (expense_loading + profit_margin)
    premium = pure_premium + total_loading

    return premium
