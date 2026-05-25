"""
Exploratory Data Analysis Utilities

This module provides functions for comprehensive EDA including visualizations,
statistical summaries, and data quality assessments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple


class EDAAnalyzer:
    """Comprehensive EDA analysis toolkit."""

    def __init__(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize EDAAnalyzer.

        Parameters
        ----------
        df : pd.DataFrame
            Data to analyze.
        figsize : Tuple[int, int], default=(12, 6)
            Default figure size for plots.
        """
        self.df = df
        self.figsize = figsize
        sns.set_style("whitegrid")

    def data_quality_report(self) -> dict:
        """
        Generate comprehensive data quality report.

        Returns
        -------
        dict
            Data quality metrics.
        """
        report = {
            "shape": self.df.shape,
            "total_cells": self.df.shape[0] * self.df.shape[1],
            "missing_cells": self.df.isnull().sum().sum(),
            "missing_percentage": (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100,
            "duplicate_rows": self.df.duplicated().sum(),
            "columns_by_type": {
                "numerical": len(self.df.select_dtypes(include=[np.number]).columns),
                "categorical": len(self.df.select_dtypes(include=["object"]).columns),
                "datetime": len(self.df.select_dtypes(include=["datetime64"]).columns),
            },
        }
        return report

    def missing_values_analysis(self) -> pd.DataFrame:
        """
        Analyze missing values by column.

        Returns
        -------
        pd.DataFrame
            Missing values summary.
        """
        missing = pd.DataFrame({
            "Column": self.df.columns,
            "Missing_Count": self.df.isnull().sum().values,
            "Missing_Percentage": (self.df.isnull().sum().values / len(self.df) * 100),
            "Data_Type": self.df.dtypes.values,
        })
        return missing.sort_values("Missing_Percentage", ascending=False)

    def plot_missing_values(self) -> None:
        """Visualize missing values by column."""
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]

        if len(missing_pct) == 0:
            print("No missing values found.")
            return

        fig, ax = plt.subplots(figsize=self.figsize)
        missing_pct.plot(kind="barh", ax=ax, color="coral")
        ax.set_xlabel("Missing Percentage (%)")
        ax.set_title("Missing Values by Column")
        plt.tight_layout()
        plt.show()

    def plot_numerical_distributions(self, columns: Optional[List[str]] = None) -> None:
        """
        Plot distributions of numerical columns.

        Parameters
        ----------
        columns : List[str], optional
            Specific columns to plot. If None, plots all numerical columns.
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if columns:
            numerical_cols = [col for col in columns if col in numerical_cols]

        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for idx, col in enumerate(numerical_cols):
            axes[idx].hist(self.df[col].dropna(), bins=50, color="skyblue", edgecolor="black")
            axes[idx].set_title(f"Distribution of {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")

        for idx in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    def plot_categorical_distributions(self, columns: Optional[List[str]] = None, top_n: int = 10) -> None:
        """
        Plot distributions of categorical columns.

        Parameters
        ----------
        columns : List[str], optional
            Specific columns to plot. If None, plots all categorical columns.
        top_n : int, default=10
            Show top N categories.
        """
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        if columns:
            categorical_cols = [col for col in columns if col in categorical_cols]

        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for idx, col in enumerate(categorical_cols):
            value_counts = self.df[col].value_counts().head(top_n)
            axes[idx].bar(range(len(value_counts)), value_counts.values, color="lightgreen", edgecolor="black")
            axes[idx].set_xticks(range(len(value_counts)))
            axes[idx].set_xticklabels(value_counts.index, rotation=45, ha="right")
            axes[idx].set_title(f"Top {top_n} Categories: {col}")
            axes[idx].set_ylabel("Count")

        for idx in range(len(categorical_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, columns: Optional[List[str]] = None) -> None:
        """
        Plot boxplots for outlier detection.

        Parameters
        ----------
        columns : List[str], optional
            Specific columns to plot. If None, plots all numerical columns.
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if columns:
            numerical_cols = [col for col in columns if col in numerical_cols]

        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for idx, col in enumerate(numerical_cols):
            axes[idx].boxplot(self.df[col].dropna())
            axes[idx].set_title(f"Boxplot: {col}")
            axes[idx].set_ylabel(col)

        for idx in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, columns: Optional[List[str]] = None) -> None:
        """
        Plot correlation matrix heatmap.

        Parameters
        ----------
        columns : List[str], optional
            Specific columns to include. If None, uses all numerical columns.
        """
        numerical_df = self.df.select_dtypes(include=[np.number])

        if columns:
            numerical_df = numerical_df[[col for col in columns if col in numerical_df.columns]]

        corr_matrix = numerical_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title("Correlation Matrix - Numerical Features")
        plt.tight_layout()
        plt.show()

    def plot_scatter(self, x_col: str, y_col: str, hue_col: Optional[str] = None) -> None:
        """
        Plot scatter plot for bivariate analysis.

        Parameters
        ----------
        x_col : str
            Column for x-axis.
        y_col : str
            Column for y-axis.
        hue_col : str, optional
            Column for color coding.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if hue_col:
            for category in self.df[hue_col].unique():
                mask = self.df[hue_col] == category
                ax.scatter(self.df[mask][x_col], self.df[mask][y_col], label=category, alpha=0.6)
            ax.legend()
        else:
            ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        plt.tight_layout()
        plt.show()

    def get_outliers(self, column: str, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """
        Identify outliers in a column.

        Parameters
        ----------
        column : str
            Column to analyze.
        method : str, default="iqr"
            Method: "iqr" or "zscore".
        threshold : float, default=1.5
            Threshold for IQR method.

        Returns
        -------
        pd.DataFrame
            Rows containing outliers.
        """
        if method == "iqr":
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        elif method == "zscore":
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[column].dropna()))
            outliers = self.df[np.abs(stats.zscore(self.df[column])) > threshold]
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

        return outliers


def get_loss_ratio_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Calculate loss ratio by group.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with TotalPremium and TotalClaims columns.
    group_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Loss ratio summary by group.
    """
    grouped = df.groupby(group_col).agg({
        "TotalPremium": "sum",
        "TotalClaims": "sum",
    }).reset_index()

    grouped["LossRatio"] = grouped["TotalClaims"] / grouped["TotalPremium"]
    grouped["Margin"] = grouped["TotalPremium"] - grouped["TotalClaims"]
    grouped["PolicyCount"] = df.groupby(group_col).size().values

    return grouped.sort_values("LossRatio", ascending=False)


def get_claim_frequency(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate claim frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_col : str, optional
        Column to group by. If None, returns overall frequency.

    Returns
    -------
    pd.DataFrame
        Claim frequency summary.
    """
    if group_col:
        freq = df.groupby(group_col).apply(
            lambda x: (x["TotalClaims"] > 0).sum() / len(x)
        ).reset_index()
        freq.columns = [group_col, "ClaimFrequency"]
        return freq
    else:
        return pd.DataFrame({
            "ClaimFrequency": [(df["TotalClaims"] > 0).sum() / len(df)]
        })
