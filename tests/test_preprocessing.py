"""Tests for data preprocessing / imputation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eda_app.data.preprocessing import impute_categorical, impute_numeric


@pytest.fixture
def df_numeric_nans() -> pd.DataFrame:
    """DataFrame with numeric missing values."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [10.0, None, 30.0, None, 50.0],
        }
    )


@pytest.fixture
def df_categorical_nans() -> pd.DataFrame:
    """DataFrame with categorical missing values."""
    return pd.DataFrame(
        {
            "color": ["red", None, "blue", "red", None],
            "size": ["S", "M", None, "L", "M"],
        }
    )


class TestImputeNumeric:
    """Tests for numeric imputation."""

    def test_mean_imputation(self, df_numeric_nans: pd.DataFrame) -> None:
        result = impute_numeric(df_numeric_nans, ["a"], "mean")
        assert result["a"].isna().sum() == 0
        assert result.loc[2, "a"] == pytest.approx(3.0, rel=0.01)

    def test_median_imputation(self, df_numeric_nans: pd.DataFrame) -> None:
        result = impute_numeric(df_numeric_nans, ["a"], "median")
        assert result["a"].isna().sum() == 0

    def test_ffill_imputation(self, df_numeric_nans: pd.DataFrame) -> None:
        result = impute_numeric(df_numeric_nans, ["a"], "ffill")
        assert result["a"].isna().sum() == 0
        assert result.loc[2, "a"] == 2.0  # forward fill from index 1

    def test_bfill_imputation(self, df_numeric_nans: pd.DataFrame) -> None:
        result = impute_numeric(df_numeric_nans, ["a"], "bfill")
        assert result["a"].isna().sum() == 0
        assert result.loc[2, "a"] == 4.0  # backward fill from index 3

    def test_drop_imputation(self, df_numeric_nans: pd.DataFrame) -> None:
        result = impute_numeric(df_numeric_nans, ["a"], "drop")
        assert result["a"].isna().sum() == 0
        assert len(result) == 4  # one row dropped

    def test_does_not_mutate_original(self, df_numeric_nans: pd.DataFrame) -> None:
        original_nans = df_numeric_nans["a"].isna().sum()
        _ = impute_numeric(df_numeric_nans, ["a"], "mean")
        assert df_numeric_nans["a"].isna().sum() == original_nans


class TestImputeCategorical:
    """Tests for categorical imputation."""

    def test_text_imputation(self, df_categorical_nans: pd.DataFrame) -> None:
        result = impute_categorical(
            df_categorical_nans,
            ["color"],
            "text",
            fill_values={"color": "unknown"},
        )
        assert result["color"].isna().sum() == 0
        assert (result["color"] == "unknown").sum() == 2

    def test_mode_imputation(self, df_categorical_nans: pd.DataFrame) -> None:
        result = impute_categorical(df_categorical_nans, ["color"], "mode")
        assert result["color"].isna().sum() == 0

    def test_drop_imputation(self, df_categorical_nans: pd.DataFrame) -> None:
        result = impute_categorical(df_categorical_nans, ["color"], "drop")
        assert result["color"].isna().sum() == 0
        assert len(result) == 3
