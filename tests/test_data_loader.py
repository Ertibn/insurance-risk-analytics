"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader, load_insurance_data


@pytest.fixture
def sample_data():
    """Create sample insurance data for testing."""
    data = {
        "PolicyID": [1, 2, 3, 4, 5],
        "TotalPremium": [1000, 1500, 2000, 1200, 1800],
        "TotalClaims": [500, 0, 1500, 600, 0],
        "Province": ["GP", "WC", "GP", "KZN", "WC"],
        "Gender": ["M", "F", "M", "F", "M"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink()


class TestDataLoader:
    """Test DataLoader class."""

    def test_load_data(self, temp_csv_file):
        """Test loading data from CSV."""
        loader = DataLoader(temp_csv_file)
        df = loader.load_data()
        assert df is not None
        assert len(df) == 5
        assert list(df.columns) == ["PolicyID", "TotalPremium", "TotalClaims", "Province", "Gender"]

    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        loader = DataLoader("nonexistent_file.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_validate_data(self, temp_csv_file):
        """Test data validation."""
        loader = DataLoader(temp_csv_file)
        loader.load_data()
        report = loader.validate_data()

        assert report["shape"] == (5, 5)
        assert "columns" in report
        assert "dtypes" in report
        assert "missing_values" in report

    def test_create_derived_metrics(self, temp_csv_file):
        """Test creation of derived metrics."""
        loader = DataLoader(temp_csv_file)
        df = loader.load_data()
        df_derived = loader.create_derived_metrics(df)

        assert "LossRatio" in df_derived.columns
        assert "Margin" in df_derived.columns
        assert "ClaimIndicator" in df_derived.columns

        # Check calculations
        assert df_derived.loc[0, "LossRatio"] == pytest.approx(500 / 1000)
        assert df_derived.loc[0, "Margin"] == 500
        assert df_derived.loc[0, "ClaimIndicator"] == 1
        assert df_derived.loc[1, "ClaimIndicator"] == 0

    def test_get_summary_statistics(self, temp_csv_file):
        """Test summary statistics."""
        loader = DataLoader(temp_csv_file)
        df = loader.load_data()
        stats = loader.get_summary_statistics(df)

        assert "TotalPremium" in stats.columns
        assert "TotalClaims" in stats.columns
        assert "count" in stats.index


class TestLoadInsuranceData:
    """Test convenience functions."""

    def test_load_insurance_data(self, temp_csv_file):
        """Test load_insurance_data function."""
        df = load_insurance_data(temp_csv_file)
        assert df is not None
        assert len(df) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
