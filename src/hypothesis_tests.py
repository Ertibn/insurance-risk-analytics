"""
Hypothesis Testing Utilities

This module provides functions for statistical hypothesis testing including
chi-squared, t-tests, and z-tests for A/B testing and risk analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional


class HypothesisTest:
    """Hypothesis testing toolkit for insurance analytics."""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize HypothesisTest.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.
        """
        self.alpha = alpha
        self.results = []

    def chi_squared_test(
        self,
        df: pd.DataFrame,
        group_col: str,
        outcome_col: str,
        test_name: str = "Chi-Squared Test",
    ) -> Dict:
        """
        Perform chi-squared test for categorical variables.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        group_col : str
            Column defining groups.
        outcome_col : str
            Binary outcome column.
        test_name : str
            Name of the test.

        Returns
        -------
        Dict
            Test results including chi2, p-value, and decision.
        """
        contingency_table = pd.crosstab(df[group_col], df[outcome_col])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        result = {
            "test_name": test_name,
            "test_type": "Chi-Squared",
            "statistic": chi2,
            "p_value": p_value,
            "dof": dof,
            "alpha": self.alpha,
            "reject_null": p_value < self.alpha,
            "contingency_table": contingency_table,
        }

        self.results.append(result)
        return result

    def t_test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        test_name: str = "Independent t-test",
        equal_var: bool = True,
    ) -> Dict:
        """
        Perform independent samples t-test.

        Parameters
        ----------
        group_a : np.ndarray
            First group data.
        group_b : np.ndarray
            Second group data.
        test_name : str
            Name of the test.
        equal_var : bool, default=True
            Assume equal variances.

        Returns
        -------
        Dict
            Test results including t-statistic, p-value, and decision.
        """
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=equal_var)

        result = {
            "test_name": test_name,
            "test_type": "t-test",
            "statistic": t_stat,
            "p_value": p_value,
            "alpha": self.alpha,
            "reject_null": p_value < self.alpha,
            "group_a_mean": np.mean(group_a),
            "group_b_mean": np.mean(group_b),
            "group_a_std": np.std(group_a),
            "group_b_std": np.std(group_b),
        }

        self.results.append(result)
        return result

    def z_test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        test_name: str = "z-test",
    ) -> Dict:
        """
        Perform z-test for proportions.

        Parameters
        ----------
        group_a : np.ndarray
            First group data (binary).
        group_b : np.ndarray
            Second group data (binary).
        test_name : str
            Name of the test.

        Returns
        -------
        Dict
            Test results including z-statistic, p-value, and decision.
        """
        n_a = len(group_a)
        n_b = len(group_b)
        p_a = np.mean(group_a)
        p_b = np.mean(group_b)
        p_pool = (np.sum(group_a) + np.sum(group_b)) / (n_a + n_b)

        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        z_stat = (p_a - p_b) / se if se > 0 else 0

        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        result = {
            "test_name": test_name,
            "test_type": "z-test",
            "statistic": z_stat,
            "p_value": p_value,
            "alpha": self.alpha,
            "reject_null": p_value < self.alpha,
            "group_a_proportion": p_a,
            "group_b_proportion": p_b,
        }

        self.results.append(result)
        return result

    def mann_whitney_u_test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        test_name: str = "Mann-Whitney U Test",
    ) -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).

        Parameters
        ----------
        group_a : np.ndarray
            First group data.
        group_b : np.ndarray
            Second group data.
        test_name : str
            Name of the test.

        Returns
        -------
        Dict
            Test results.
        """
        u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

        result = {
            "test_name": test_name,
            "test_type": "Mann-Whitney U",
            "statistic": u_stat,
            "p_value": p_value,
            "alpha": self.alpha,
            "reject_null": p_value < self.alpha,
        }

        self.results.append(result)
        return result

    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all tests performed.

        Returns
        -------
        pd.DataFrame
            Summary table of test results.
        """
        summary_data = []
        for result in self.results:
            summary_data.append({
                "Test Name": result["test_name"],
                "Test Type": result["test_type"],
                "Statistic": result["statistic"],
                "p-value": result["p_value"],
                "Alpha": result["alpha"],
                "Reject H0": result["reject_null"],
            })

        return pd.DataFrame(summary_data)


def test_risk_by_province(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """
    Test if there are risk differences across provinces.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with Province and TotalClaims columns.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Dict
        Test results.
    """
    tester = HypothesisTest(alpha=alpha)

    # Create binary outcome: claim occurred or not
    df_test = df.copy()
    df_test["HasClaim"] = (df_test["TotalClaims"] > 0).astype(int)

    result = tester.chi_squared_test(
        df_test,
        "Province",
        "HasClaim",
        test_name="Risk Differences Across Provinces",
    )

    return result


def test_risk_by_zipcode(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """
    Test if there are risk differences across zip codes.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with ZipCode and TotalClaims columns.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Dict
        Test results.
    """
    tester = HypothesisTest(alpha=alpha)

    df_test = df.copy()
    df_test["HasClaim"] = (df_test["TotalClaims"] > 0).astype(int)

    result = tester.chi_squared_test(
        df_test,
        "ZipCode",
        "HasClaim",
        test_name="Risk Differences Across Zip Codes",
    )

    return result


def test_margin_by_zipcode(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """
    Test if there are margin differences across zip codes.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with ZipCode, TotalPremium, and TotalClaims columns.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Dict
        Test results.
    """
    tester = HypothesisTest(alpha=alpha)

    df_test = df.copy()
    df_test["Margin"] = df_test["TotalPremium"] - df_test["TotalClaims"]

    # Select top 2 zip codes for comparison
    top_zipcodes = df_test["ZipCode"].value_counts().head(2).index
    group_a = df_test[df_test["ZipCode"] == top_zipcodes[0]]["Margin"].values
    group_b = df_test[df_test["ZipCode"] == top_zipcodes[1]]["Margin"].values

    result = tester.t_test(
        group_a,
        group_b,
        test_name=f"Margin Differences: {top_zipcodes[0]} vs {top_zipcodes[1]}",
    )

    return result


def test_risk_by_gender(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """
    Test if there are risk differences between genders.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with Gender and TotalClaims columns.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Dict
        Test results.
    """
    tester = HypothesisTest(alpha=alpha)

    df_test = df.copy()
    df_test["HasClaim"] = (df_test["TotalClaims"] > 0).astype(int)

    result = tester.chi_squared_test(
        df_test,
        "Gender",
        "HasClaim",
        test_name="Risk Differences by Gender",
    )

    return result


def calculate_effect_size(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Parameters
    ----------
    group_a : np.ndarray
        First group data.
    group_b : np.ndarray
        Second group data.

    Returns
    -------
    float
        Cohen's d effect size.
    """
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_std if pooled_std > 0 else 0

    return cohens_d


def calculate_power(effect_size: float, alpha: float = 0.05, n: int = 100) -> float:
    """
    Estimate statistical power (simplified).

    Parameters
    ----------
    effect_size : float
        Cohen's d effect size.
    alpha : float, default=0.05
        Significance level.
    n : int, default=100
        Sample size per group.

    Returns
    -------
    float
        Estimated power.
    """
    from scipy.stats import nct

    # Non-centrality parameter
    lambda_param = effect_size * np.sqrt(n / 2)

    # Critical value
    t_crit = stats.t.ppf(1 - alpha / 2, 2 * n - 2)

    # Power
    power = 1 - nct.cdf(t_crit, 2 * n - 2, lambda_param) + nct.cdf(-t_crit, 2 * n - 2, lambda_param)

    return power
