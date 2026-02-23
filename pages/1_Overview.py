"""Page 1: Overview — Dataset statistics and missing values."""

from __future__ import annotations

import streamlit as st

from eda_app.components.download import add_csv_download
from eda_app.stats import compute_dataset_info, compute_summary_stats

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")


def render() -> None:
    """Render the Overview page."""
    st.header("📊 Overview — Dataset Statistics")

    if "df" not in st.session_state:
        st.warning("⬅️ Please upload a file on the Home page first.")
        return

    df = st.session_state["df"]

    # Summary statistics
    num_summary, cat_summary = compute_summary_stats(df)

    tab_num, tab_cat, tab_info = st.tabs(["📐 Numerical", "🏷️ Categorical", "ℹ️ Data Info"])

    with tab_num:
        if num_summary is not None:
            st.subheader("Numerical Summary")
            st.dataframe(num_summary, use_container_width=True)
            add_csv_download(num_summary, "numerical_summary")
        else:
            st.info("No numeric columns found.")

    with tab_cat:
        if cat_summary is not None:
            st.subheader("Categorical Summary")
            st.dataframe(cat_summary, use_container_width=True)
            add_csv_download(cat_summary, "categorical_summary")
        else:
            st.info("No categorical columns found.")

    with tab_info:
        st.subheader("Column Information")
        df_info = compute_dataset_info(df)
        st.dataframe(df_info, use_container_width=True)
        add_csv_download(df_info.reset_index(), "dataset_info")

    # Missing values section
    if df.isna().any().any():
        st.subheader("🕳️ Missing Values")
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        missing_df = missing.to_frame("Count")
        missing_df["Percentage"] = (missing / len(df) * 100).round(2)
        st.dataframe(missing_df, use_container_width=True)


render()
