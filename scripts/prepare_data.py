"""
Data Preparation Script

This script handles data loading, cleaning, and preprocessing for the insurance
risk analytics pipeline. It's designed to be run as part of the DVC pipeline.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader


def prepare_data(input_path: str, output_path: str, report_path: str) -> None:
    """
    Prepare data for analysis.

    Parameters
    ----------
    input_path : str
        Path to raw data CSV.
    output_path : str
        Path to save cleaned data.
    report_path : str
        Path to save data quality report.
    """
    print(f"Loading data from {input_path}...")
    loader = DataLoader(input_path)

    # Load data
    df = loader.load_data()
    print(f"Data loaded: {df.shape}")

    # Validate data
    validation_report = loader.validate_data()
    print(f"Data validation complete")

    # Handle missing values
    df_clean = loader.handle_missing_values(strategy="impute", threshold=0.5)
    print(f"Missing values handled: {df_clean.shape}")

    # Create derived metrics
    df_clean = loader.create_derived_metrics(df_clean)
    print(f"Derived metrics created")

    # Save cleaned data
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

    # Generate quality report
    quality_report = {
        "original_shape": validation_report["shape"],
        "cleaned_shape": df_clean.shape,
        "rows_removed": validation_report["shape"][0] - df_clean.shape[0],
        "missing_percentage": {k: float(v) for k, v in validation_report["missing_percentage"].items()},
        "duplicate_rows": 0,
        "columns": validation_report["columns"],
        "derived_metrics": ["LossRatio", "Margin", "ClaimIndicator"],
    }

    # Save report
    report_dir = Path(report_path).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(quality_report, f, indent=2)
    print(f"Quality report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Original shape: {validation_report['shape']}")
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Rows removed: {quality_report['rows_removed']}")
    print(f"Columns: {df_clean.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    # Default paths
    input_path = "data/insurance_data.csv"
    output_path = "data/processed/cleaned_data.csv"
    report_path = "data/processed/data_quality_report.json"

    # Allow command-line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        report_path = sys.argv[3]

    prepare_data(input_path, output_path, report_path)
