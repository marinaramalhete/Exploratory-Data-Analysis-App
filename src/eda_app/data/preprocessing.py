"""Data preprocessing and missing value imputation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    import pandas as pd


def impute_numeric(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
) -> pd.DataFrame:
    """Impute missing values in numeric columns.

    Args:
        df: Input DataFrame.
        columns: Column names to impute.
        method: Imputation strategy. One of 'mean', 'median', 'mode',
            'ffill', 'bfill', 'drop'.

    Returns:
        DataFrame with missing values handled according to the chosen method.
    """
    result = df.copy()

    match method:
        case "mean":
            for col in columns:
                fill_value = df[col].mean()
                st.caption(f"Mean of `{col}`: {fill_value:.4f}")
                result[col] = df[col].fillna(fill_value)

        case "median":
            for col in columns:
                fill_value = df[col].median()
                st.caption(f"Median of `{col}`: {fill_value:.4f}")
                result[col] = df[col].fillna(fill_value)

        case "mode":
            for col in columns:
                fill_value = df[col].mode()[0]
                st.caption(f"Mode of `{col}`: {fill_value}")
                result[col] = df[col].fillna(fill_value)

        case "ffill":
            result[columns] = df[columns].ffill()

        case "bfill":
            result[columns] = df[columns].bfill()

        case "drop":
            result = df.dropna(subset=columns)
            st.caption(
                f"Rows: {df.shape[0]} → {result.shape[0]} ({df.shape[0] - result.shape[0]} dropped)"
            )

    return result


def impute_categorical(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    fill_values: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Impute missing values in categorical columns.

    Args:
        df: Input DataFrame.
        columns: Column names to impute.
        method: Imputation strategy. One of 'text', 'mode', 'drop'.
        fill_values: Mapping of column name → fill text (used with 'text' method).

    Returns:
        DataFrame with missing values handled.
    """
    result = df.copy()

    match method:
        case "text":
            fill_values = fill_values or {}
            for col in columns:
                result[col] = df[col].fillna(fill_values.get(col, "Unknown"))

        case "mode":
            for col in columns:
                fill_value = df[col].mode()[0]
                st.caption(f"Mode of `{col}`: {fill_value}")
                result[col] = df[col].fillna(fill_value)

        case "drop":
            result = df.dropna(subset=columns)
            st.caption(
                f"Rows: {df.shape[0]} → {result.shape[0]} ({df.shape[0] - result.shape[0]} dropped)"
            )

    return result
