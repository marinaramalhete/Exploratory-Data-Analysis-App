"""EDA App — Main entry point and home page.

This is the Streamlit multi-page app entry point. It handles file upload
and stores the DataFrame in session state for use across all pages.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="EDA App — Exploratory Data Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Render the home page with file upload and instructions."""
    st.title("🔍 Exploratory Data Analysis App")
    st.markdown(
        "Upload your dataset to get started with comprehensive statistical analysis "
        "and interactive visualizations."
    )

    # --- File Upload ---
    st.sidebar.header("📂 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Supported formats: CSV, Excel (.xlsx), Parquet",
    )

    if uploaded_file is not None:
        try:
            from eda_app.data import load_file

            df = load_file(uploaded_file, uploaded_file.name)
            st.session_state["df"] = df
            st.session_state["filename"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

        st.success(
            f"✅ Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns"
        )

        # Quick overview metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", f"{df.shape[1]:,}")
        missing_pct = df.isna().sum().sum() / df.size * 100
        col3.metric("Missing %", f"{missing_pct:.2f}%")
        col4.metric("Duplicates", f"{df.duplicated().sum():,}")

        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.info("👈 Use the **sidebar pages** to explore your data in detail.")

    elif "df" in st.session_state:
        df = st.session_state["df"]
        filename = st.session_state.get("filename", "data")
        st.success(
            f"✅ Using previously loaded **{filename}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns"
        )
        st.dataframe(df.head(10), use_container_width=True)
        st.info("👈 Use the **sidebar pages** to explore your data in detail.")

    else:
        st.markdown("---")
        st.markdown(
            """
            ### Getting Started

            1. **Upload** a CSV, Excel, or Parquet file using the sidebar
            2. **Overview** — view data types, missing values, and summary statistics
            3. **Univariate** — analyze individual variables with histograms, boxplots, and distribution plots
            4. **Multivariate** — explore relationships between variables with scatter plots, correlation heatmaps, and more
            5. **Profiling** — generate a comprehensive automated report with alerts and outlier detection

            ---

            *Built with [Streamlit](https://streamlit.io) • [GitHub](https://github.com/marinaramalhete)*
            """
        )

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**EDA App v1.0** · Built by [Marina Ramalhete](https://linkedin.com/in/marinaramalhete)"
    )


if __name__ == "__main__":
    main()
else:
    main()
