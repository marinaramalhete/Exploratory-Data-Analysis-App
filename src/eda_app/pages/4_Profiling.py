"""Page 4: Profiling — Automated comprehensive data report."""

from __future__ import annotations

import streamlit as st

from eda_app.stats.profiling import render_profiling

st.set_page_config(page_title="Auto Profiling", page_icon="📋", layout="wide")


def render() -> None:
    """Render the Profiling page."""
    st.header("📋 Automated Data Profiling")
    st.markdown(
        "Comprehensive automated report with distribution analysis, "
        "correlation matrix, outlier detection, and data quality alerts."
    )

    if "df" not in st.session_state:
        st.warning("⬅️ Please upload a file on the Home page first.")
        return

    df = st.session_state["df"]

    with st.spinner("Generating profiling report..."):
        render_profiling(df)


render()
