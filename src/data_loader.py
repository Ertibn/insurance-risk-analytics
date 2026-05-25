"""
Data Loading and Preprocessing Utilities

This module provides functions for loading, validating, and preprocessing
insurance claim data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Load and preprocess insurance claim data."""

    def __init__(self, data_path: str):
        """
        Initialize DataLoader.

        Parameters
        ----------
        data_path : str
            Path to the insurance data CSV file.
        """
        self.data_path = Path(data_path)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load insurance data from CSV.

        Returns
        -------
        pd.DataFrame
            Loaded insurance data.

        Raises
        ------
        FileNotFoundError
            If data file does not exist.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        return self.df

    def validate_data(self) -> dict:
        """
        Validate data integrity and structure.

        Returns
        -------
        dict
            Validation report with shape, dtypes, and missing values.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        report = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
        }
        return report

    def handle_missing_values(
        self, strategy: str = "drop", threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Parameters
        ----------
        strategy : str, default="drop"
            Strategy for handling missing values: "drop" or "impute".
        threshold : float, default=0.5
            Drop columns with missing percentage > threshold.

        Returns
        -------
        pd.DataFrame
            Data with missing values handled.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.df.copy()

        # Drop columns with too many missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index
        df = df.drop(columns=cols_to_drop)

        if strategy == "drop":
            df = df.dropna()
        elif strategy == "impute":
            # Impute numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

            # Impute categorical columns with mode
            categorical_cols = df.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

        return df

    def create_derived_metrics(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create derived metrics for analysis.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame to process. If None, uses self.df.

        Returns
        -------
        pd.DataFrame
            DataFrame with derived metrics added.
        """
        if df is None:
            if self.df is None:
                raise ValueError("Data not loaded. Call load_data() first.")
            df = self.df.copy()
        else:
            df = df.copy()

        # Create derived metrics
        if "TotalPremium" in df.columns and "TotalClaims" in df.columns:
            df["LossRatio"] = df["TotalClaims"] / (df["TotalPremium"] + 1e-6)
            df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
            df["ClaimIndicator"] = (df["TotalClaims"] > 0).astype(int)

        return df

    def get_summary_statistics(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get summary statistics for numerical columns.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame to summarize. If None, uses self.df.

        Returns
        -------
        pd.DataFrame
            Summary statistics.
        """
        if df is None:
            if self.df is None:
                raise ValueError("Data not loaded. Call load_data() first.")
            df = self.df
        
        return df.describe()


def load_insurance_data(data_path: str) -> pd.DataFrame:
    """
    Convenience function to load insurance data.

    Parameters
    ----------
    data_path : str
        Path to the insurance data CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded insurance data.
    """
    loader = DataLoader(data_path)
    return loader.load_data()


def prepare_data_for_modeling(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for modeling by splitting into train/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_col : str
        Name of target column.
    test_size : float, default=0.2
        Proportion of data for testing.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
