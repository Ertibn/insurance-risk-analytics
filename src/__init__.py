"""
Insurance Risk Analytics Package

A comprehensive toolkit for analyzing insurance claim data, building predictive models,
and developing risk-based pricing strategies.
"""

__version__ = "0.1.0"
__author__ = "10 Academy Data Analytics Team"

from . import data_loader
from . import eda_utils
from . import hypothesis_tests
from . import modeling

__all__ = [
    "data_loader",
    "eda_utils",
    "hypothesis_tests",
    "modeling",
]
