"""Data loading utilities for the EDA app."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from io import BytesIO


@st.cache_data
def load_file(file: BytesIO, filename: str) -> pd.DataFrame:
    """Load a data file into a pandas DataFrame.

    Supports CSV, Excel (.xlsx/.xls), and Parquet formats.
    The file type is inferred from the filename extension.

    Args:
        file: The uploaded file buffer.
        filename: Original filename to determine format.

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        ValueError: If the file format is not supported.
    """
    suffix = Path(filename).suffix.lower()

    match suffix:
        case ".csv":
            return pd.read_csv(file)
        case ".xlsx" | ".xls":
            return pd.read_excel(file, engine="openpyxl")
        case ".parquet":
            return pd.read_parquet(file)
        case _:
            msg = f"Unsupported file format: {suffix}"
            raise ValueError(msg)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric column names.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names with numeric dtypes.
    """
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return list of categorical (object/string) column names.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names with object/string dtypes.
    """
    return df.select_dtypes(include=["object", "category"]).columns.tolist()
