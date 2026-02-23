"""Page 2: Univariate — Single variable analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from src.eda_app.components.download import add_matplotlib_download, add_plotly_download
from src.eda_app.data import get_categorical_columns, get_numeric_columns
from src.eda_app.stats import (
    compute_descriptive_stats,
    compute_quantile_stats,
    compute_variable_info,
)
from src.eda_app.visualization import EDAPlotter

st.set_page_config(page_title="Univariate Analysis", page_icon="📈", layout="wide")


def render() -> None:
    """Render the Univariate Analysis page."""
    st.header("📈 Univariate Analysis")
    st.markdown("Analyze individual variables with descriptive statistics and charts.")

    if "df" not in st.session_state:
        st.warning("⬅️ Please upload a file on the Home page first.")
        return

    df = st.session_state["df"]
    plotter = EDAPlotter(df)
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    all_cols = list(df.columns)

    main_var = st.selectbox(
        "Choose a variable to analyze:",
        options=["", *all_cols],
        key="uni_main_var",
    )

    if not main_var:
        st.info("Select a variable above to begin analysis.")
        return

    # --- Numeric variable ---
    if main_var in numeric_cols:
        # Statistics tabs
        tab_info, tab_desc, tab_quant = st.tabs(
            ["ℹ️ Variable Info", "📐 Descriptive Stats", "📊 Quantiles"]
        )

        with tab_info:
            st.dataframe(
                compute_variable_info(df, [main_var]).T,
                use_container_width=True,
            )

        with tab_desc:
            st.dataframe(
                compute_descriptive_stats(df, [main_var]).T,
                use_container_width=True,
            )

        with tab_quant:
            st.dataframe(
                compute_quantile_stats(df, [main_var]).T,
                use_container_width=True,
            )

        # Chart options
        st.subheader("📉 Charts")
        chart_type = st.radio(
            "Chart type",
            ["Histogram", "BoxPlot", "Distribution Plot"],
            horizontal=True,
            key="uni_chart_type",
        )

        if chart_type == "Histogram":
            col_opts, col_chart = st.columns([1, 3])
            with col_opts:
                hue = st.selectbox(
                    "Color by (optional):",
                    [None, *all_cols],
                    key="uni_hist_hue",
                )
                nbins = st.slider("Number of bins", 5, 200, 50, key="uni_hist_bins")
                vmin, vmax = float(df[main_var].min()), float(df[main_var].max())
                range_values = st.slider(
                    "Value range",
                    vmin,
                    vmax,
                    (vmin, vmax),
                    key="uni_hist_range",
                )
            with col_chart:
                fig = plotter.histogram(main_var, color=hue, nbins=nbins, range_values=range_values)
                st.plotly_chart(fig, use_container_width=True)
                add_plotly_download(fig, f"histogram_{main_var}")

        elif chart_type == "BoxPlot":
            col_opts, col_chart = st.columns([1, 3])
            with col_opts:
                col_x = st.selectbox("X variable (optional):", [None, *all_cols], key="uni_box_x")
                hue = st.selectbox("Color by (optional):", [None, *all_cols], key="uni_box_hue")
            with col_chart:
                fig = plotter.box_plot(main_var, x=col_x, color=hue)
                st.plotly_chart(fig, use_container_width=True)
                add_plotly_download(fig, f"boxplot_{main_var}")

        elif chart_type == "Distribution Plot":
            fig = plotter.dist_plot(main_var)
            st.pyplot(fig)
            add_matplotlib_download(fig, f"dist_{main_var}")
            plt.close(fig)

    # --- Categorical variable ---
    elif main_var in categorical_cols:
        desc = df[main_var].describe().to_frame()
        st.dataframe(desc, use_container_width=True)

        value_counts = df[main_var].value_counts()
        st.bar_chart(value_counts)

    # --- Sidebar: explore category frequencies ---
    st.sidebar.subheader("🔎 Quick Category Explorer")
    explore_var = st.sidebar.selectbox(
        "Check unique values & frequency:",
        ["", *all_cols],
        key="uni_explore_var",
    )
    if explore_var:
        freq = df[explore_var].value_counts(dropna=False).to_frame("Count")
        freq["Percentage"] = (freq["Count"] / len(df) * 100).round(2)
        st.sidebar.dataframe(freq, use_container_width=True)


render()
