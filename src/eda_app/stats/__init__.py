"""Descriptive statistics computations."""

from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data
def compute_descriptive_stats(_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns.

    Calculates mean, std, variance, kurtosis, skewness,
    and coefficient of variation.

    Args:
        _df: Input DataFrame (underscore prefix for Streamlit cache hashing).
        columns: Numeric column names to analyze.

    Returns:
        DataFrame with statistics as rows, variables as columns.
    """
    stats: dict[str, pd.Series] = {
        "Mean": _df[columns].mean(),
        "Std": _df[columns].std(),
        "Variance": _df[columns].var(),
        "Kurtosis": _df[columns].kurtosis(),
        "Skewness": _df[columns].skew(),
    }
    stats["CV (Std/Mean)"] = stats["Std"] / stats["Mean"]
    return pd.DataFrame(stats, index=columns).T.round(4)


@st.cache_data
def compute_quantile_stats(_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute quantile statistics for numeric columns.

    Calculates min, Q1, median, Q3, max, range, and IQR.

    Args:
        _df: Input DataFrame.
        columns: Numeric column names to analyze.

    Returns:
        DataFrame with quantile statistics.
    """
    stats_q: dict[str, pd.Series] = {
        "Min": _df[columns].min(),
        "Q1": _df[columns].quantile(0.25),
        "Median": _df[columns].quantile(0.50),
        "Q3": _df[columns].quantile(0.75),
        "Max": _df[columns].max(),
    }
    stats_q["Range"] = stats_q["Max"] - stats_q["Min"]
    stats_q["IQR"] = stats_q["Q3"] - stats_q["Q1"]
    return pd.DataFrame(stats_q, index=columns).T.round(4)


@st.cache_data
def compute_variable_info(_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute metadata/info for selected columns.

    Includes dtype, unique count, zero count/percentage,
    and missing count/percentage.

    Args:
        _df: Input DataFrame.
        columns: Column names to analyze.

    Returns:
        DataFrame with variable information.
    """
    zeros = (_df[columns] == 0).sum()
    zeros_pct = (zeros * 100 / len(_df)).round(2)
    info: dict[str, pd.Series] = {
        "Type": _df[columns].dtypes,
        "Unique": _df[columns].nunique(),
        "Zeros": zeros,
        "Zeros %": zeros_pct,
        "Missing": _df[columns].isna().sum(),
        "Missing %": (_df[columns].isna().sum() / _df.shape[0] * 100).round(2),
    }
    return pd.DataFrame(info, index=columns).T


@st.cache_data
def compute_dataset_info(_df: pd.DataFrame) -> pd.DataFrame:
    """Compute overview info for the entire dataset.

    Args:
        _df: Input DataFrame.

    Returns:
        DataFrame with types, NaN counts, NaN %, and unique counts per column.
    """
    return pd.DataFrame(
        {
            "Type": _df.dtypes,
            "Missing": _df.isna().sum(),
            "Missing %": (_df.isna().sum() / len(_df) * 100).round(2),
            "Unique": _df.nunique(),
        }
    )


@st.cache_data
def compute_summary_stats(_df: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Compute summary statistics split by numeric and categorical columns.

    Args:
        _df: Input DataFrame.

    Returns:
        Tuple of (numeric_summary, categorical_summary). Either may be None
        if no columns of that type exist.
    """
    numeric_df = _df.select_dtypes(include="number")
    categorical_df = _df.select_dtypes(include=["object", "category"])

    num_summary = numeric_df.describe().T if not numeric_df.empty else None
    cat_summary = categorical_df.describe().T if not categorical_df.empty else None

    return num_summary, cat_summary
