"""Tests for descriptive statistics functions."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eda_app.stats import (
    compute_dataset_info,
    compute_descriptive_stats,
    compute_quantile_stats,
    compute_summary_stats,
    compute_variable_info,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with mixed types."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 50],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "city": ["NYC", "LA", "NYC", "LA", "NYC"],
        }
    )


@pytest.fixture
def df_with_nans() -> pd.DataFrame:
    """Create a DataFrame with missing values."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0, 5.0],
            "y": [10.0, None, 30.0, None, 50.0],
            "cat": ["a", None, "b", "a", None],
        }
    )


class TestDescriptiveStats:
    """Tests for compute_descriptive_stats."""

    def test_basic_stats(self, sample_df: pd.DataFrame) -> None:
        compute_descriptive_stats.clear()
        result = compute_descriptive_stats(sample_df, ["age", "salary"])
        assert "Mean" in result.index
        assert "Std" in result.index
        assert "Kurtosis" in result.index
        assert "Skewness" in result.index
        assert result.loc["Mean", "age"] == pytest.approx(36.0, rel=0.01)

    def test_single_column(self, sample_df: pd.DataFrame) -> None:
        compute_descriptive_stats.clear()
        result = compute_descriptive_stats(sample_df, ["salary"])
        assert result.shape[1] == 1
        assert "salary" in result.columns


class TestQuantileStats:
    """Tests for compute_quantile_stats."""

    def test_quantiles(self, sample_df: pd.DataFrame) -> None:
        compute_quantile_stats.clear()
        result = compute_quantile_stats(sample_df, ["age"])
        assert "Min" in result.index
        assert "Q1" in result.index
        assert "Median" in result.index
        assert "Q3" in result.index
        assert "Max" in result.index
        assert "IQR" in result.index
        assert result.loc["Min", "age"] == 25
        assert result.loc["Max", "age"] == 50


class TestVariableInfo:
    """Tests for compute_variable_info."""

    def test_info_columns(self, df_with_nans: pd.DataFrame) -> None:
        compute_variable_info.clear()
        result = compute_variable_info(df_with_nans, ["x", "y"])
        assert "Missing" in result.index
        assert "Missing %" in result.index
        assert "Unique" in result.index


class TestDatasetInfo:
    """Tests for compute_dataset_info."""

    def test_dataset_info(self, sample_df: pd.DataFrame) -> None:
        compute_dataset_info.clear()
        result = compute_dataset_info(sample_df)
        assert len(result) == 4  # 4 columns
        assert "Type" in result.columns
        assert "Missing" in result.columns


class TestSummaryStats:
    """Tests for compute_summary_stats."""

    def test_both_types(self, sample_df: pd.DataFrame) -> None:
        compute_summary_stats.clear()
        num_summary, cat_summary = compute_summary_stats(sample_df)
        assert num_summary is not None
        assert cat_summary is not None
        assert "age" in num_summary.index
        assert "name" in cat_summary.index

    def test_numeric_only(self) -> None:
        compute_summary_stats.clear()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        num_summary, cat_summary = compute_summary_stats(df)
        assert num_summary is not None
        assert cat_summary is None
