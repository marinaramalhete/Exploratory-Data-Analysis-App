"""Tests for data loading utilities."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.eda_app.data import get_categorical_columns, get_numeric_columns, load_file


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, None],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "name": ["Alice", "Bob", "Charlie", "Diana", None],
            "city": ["NYC", "LA", "NYC", "LA", "NYC"],
        }
    )


@pytest.fixture
def csv_buffer(sample_df: pd.DataFrame) -> io.BytesIO:
    """Create a CSV file buffer from sample data."""
    buf = io.BytesIO()
    sample_df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


@pytest.fixture
def excel_buffer(sample_df: pd.DataFrame) -> io.BytesIO:
    """Create an Excel file buffer from sample data."""
    buf = io.BytesIO()
    sample_df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf


@pytest.fixture
def parquet_buffer(sample_df: pd.DataFrame) -> io.BytesIO:
    """Create a Parquet file buffer from sample data."""
    buf = io.BytesIO()
    sample_df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf


class TestLoadFile:
    """Tests for the load_file function."""

    def test_load_csv(self, csv_buffer: io.BytesIO) -> None:
        load_file.clear()
        df = load_file(csv_buffer, "test.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 4)

    def test_load_excel(self, excel_buffer: io.BytesIO) -> None:
        load_file.clear()
        df = load_file(excel_buffer, "test.xlsx")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 4)

    def test_load_parquet(self, parquet_buffer: io.BytesIO) -> None:
        load_file.clear()
        df = load_file(parquet_buffer, "test.parquet")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 4)

    def test_unsupported_format(self) -> None:
        load_file.clear()
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_file(io.BytesIO(b"data"), "test.json")


class TestColumnDetection:
    """Tests for column type detection functions."""

    def test_get_numeric_columns(self, sample_df: pd.DataFrame) -> None:
        result = get_numeric_columns(sample_df)
        assert "age" in result
        assert "salary" in result
        assert "name" not in result

    def test_get_categorical_columns(self, sample_df: pd.DataFrame) -> None:
        result = get_categorical_columns(sample_df)
        assert "name" in result
        assert "city" in result
        assert "age" not in result
