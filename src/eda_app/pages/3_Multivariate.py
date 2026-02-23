"""Page 3: Multivariate — Multi-variable analysis and charts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from eda_app.components.download import add_matplotlib_download, add_plotly_download
from eda_app.visualization import EDAPlotter

st.set_page_config(page_title="Multivariate Analysis", page_icon="📉", layout="wide")


def _format_aggfunc(func: str) -> str:
    """Format aggregation function for display."""
    return func.capitalize()


def render() -> None:
    """Render the Multivariate Analysis page."""
    st.header("📉 Multivariate Analysis")
    st.markdown("Explore relationships between variables with interactive charts.")

    if "df" not in st.session_state:
        st.warning("⬅️ Please upload a file on the Home page first.")
        return

    df = st.session_state["df"]
    plotter = EDAPlotter(df)
    all_cols = list(df.columns)
    num_cols = list(plotter.num_vars)
    col_with_none: list[str | None] = [None, *all_cols]

    # Chart type selection
    chart_type = st.sidebar.radio(
        "📊 Chart type",
        [
            "Correlation",
            "Boxplot",
            "Violin",
            "Swarmplot",
            "Heatmap",
            "Histogram",
            "Scatterplot",
            "Countplot",
            "Barplot",
            "Lineplot",
        ],
        key="mv_chart_type",
    )

    match chart_type:
        case "Correlation":
            st.subheader("🔗 Correlation Heatmap")
            method = st.sidebar.selectbox(
                "Method",
                ["pearson", "kendall", "spearman"],
                key="mv_corr_method",
            )
            selected = st.sidebar.multiselect(
                "Select columns (empty = all numeric)",
                all_cols,
                key="mv_corr_cols",
            )
            fig = plotter.correlation_heatmap(
                columns=selected if selected else None,
                method=method,
            )
            st.pyplot(fig)
            add_matplotlib_download(fig, "correlation_heatmap")
            plt.close(fig)

        case "Boxplot":
            st.subheader("📦 Boxplot")
            col_y = st.sidebar.selectbox("Y variable (numeric)", num_cols, key="mv_box_y")
            col_x = st.sidebar.selectbox("X variable (optional)", col_with_none, key="mv_box_x")
            hue = st.sidebar.selectbox("Color by (optional)", col_with_none, key="mv_box_hue")
            fig = plotter.box_plot(col_y, x=col_x, color=hue)
            st.plotly_chart(fig, use_container_width=True)
            add_plotly_download(fig, "boxplot")

        case "Violin":
            st.subheader("🎻 Violin Plot")
            col_y = st.sidebar.selectbox("Y variable (numeric)", num_cols, key="mv_vio_y")
            col_x = st.sidebar.selectbox("X variable (optional)", col_with_none, key="mv_vio_x")
            hue = st.sidebar.selectbox("Color by (optional)", col_with_none, key="mv_vio_hue")
            split = st.sidebar.checkbox("Split", key="mv_vio_split")
            fig = plotter.violin_plot(col_y, x=col_x, hue=hue, split=split)
            st.pyplot(fig)
            add_matplotlib_download(fig, "violin")
            plt.close(fig)

        case "Swarmplot":
            st.subheader("🐝 Swarm Plot")
            col_y = st.sidebar.selectbox("Y variable (numeric)", num_cols, key="mv_swarm_y")
            col_x = st.sidebar.selectbox("X variable (optional)", col_with_none, key="mv_swarm_x")
            hue = st.sidebar.selectbox("Color by (optional)", col_with_none, key="mv_swarm_hue")
            dodge = st.sidebar.checkbox("Dodge", key="mv_swarm_dodge")
            fig = plotter.swarm_plot(col_y, x=col_x, hue=hue, dodge=dodge)
            st.pyplot(fig)
            add_matplotlib_download(fig, "swarmplot")
            plt.close(fig)

        case "Heatmap":
            st.subheader("🗺️ Pivot Heatmap")
            st.caption(
                "Select 3 variables: row (categorical), column (categorical), value (numeric)."
            )
            selected = st.sidebar.multiselect(
                "Select 3 variables",
                all_cols,
                max_selections=3,
                key="mv_heat_cols",
            )
            aggfunc = st.sidebar.selectbox(
                "Aggregation",
                ["mean", "sum", "median"],
                format_func=_format_aggfunc,
                key="mv_heat_agg",
            )
            if len(selected) == 3:
                row_var, col_var, val_var = selected
                if df[val_var].dtype.kind not in ("i", "f"):
                    st.error(
                        f"**{val_var}** is not a numeric column. "
                        "The 3rd variable must be numeric so it can be aggregated "
                        f"(e.g. {aggfunc}). Please reorder your selection: "
                        "1st = row, 2nd = column, 3rd = numeric value."
                    )
                else:
                    try:
                        fig = plotter.heatmap_pivot(row_var, col_var, val_var, aggfunc)
                        st.pyplot(fig)
                        add_matplotlib_download(fig, "heatmap_pivot")
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Could not create pivot heatmap: {e}")
            else:
                st.info("Please select exactly 3 variables.")

        case "Histogram":
            st.subheader("📊 Histogram")
            col_hist = st.sidebar.selectbox("Variable (numeric)", num_cols, key="mv_hist_var")
            hue = st.sidebar.selectbox("Color by (optional)", col_with_none, key="mv_hist_hue")
            nbins = st.sidebar.slider("Bins", 5, 200, 30, key="mv_hist_bins")
            vmin, vmax = float(df[col_hist].min()), float(df[col_hist].max())
            range_values = st.sidebar.slider(
                "Range",
                vmin,
                vmax,
                (vmin, vmax),
                key="mv_hist_range",
            )
            fig = plotter.histogram(col_hist, color=hue, nbins=nbins, range_values=range_values)
            st.plotly_chart(fig, use_container_width=True)
            add_plotly_download(fig, "histogram")

        case "Scatterplot":
            st.subheader("🔵 Scatter Plot")
            col_x = st.sidebar.selectbox("X variable (numeric)", num_cols, key="mv_scat_x")
            col_y = st.sidebar.selectbox("Y variable (numeric)", num_cols, key="mv_scat_y")
            hue = st.sidebar.selectbox("Color by (optional)", col_with_none, key="mv_scat_hue")
            size = st.sidebar.selectbox("Size by (optional)", col_with_none, key="mv_scat_size")
            fig = plotter.scatter_plot(col_x, col_y, color=hue, size=size)
            st.plotly_chart(fig, use_container_width=True)
            add_plotly_download(fig, "scatterplot")

        case "Countplot":
            st.subheader("📋 Count Plot")
            col_main = st.sidebar.selectbox("Variable", all_cols, key="mv_cnt_var")
            hue = st.sidebar.selectbox("Color by (optional)", col_with_none, key="mv_cnt_hue")
            fig = plotter.count_plot(col_main, hue=hue)
            st.pyplot(fig)
            add_matplotlib_download(fig, "countplot")
            plt.close(fig)

        case "Barplot":
            st.subheader("📊 Bar Plot")
            col_y = st.sidebar.selectbox("Y variable (numeric)", num_cols, key="mv_bar_y")
            col_x = st.sidebar.selectbox("X variable", all_cols, key="mv_bar_x")
            hue = st.sidebar.selectbox("Color by (optional)", col_with_none, key="mv_bar_hue")
            fig = plotter.bar_plot(col_x, col_y, color=hue)
            st.plotly_chart(fig, use_container_width=True)
            add_plotly_download(fig, "barplot")

        case "Lineplot":
            st.subheader("📈 Line Plot")
            col_y = st.sidebar.selectbox("Y variable (numeric)", num_cols, key="mv_line_y")
            col_x = st.sidebar.selectbox("X variable", all_cols, key="mv_line_x")
            hue = st.sidebar.selectbox("Color by (optional)", col_with_none, key="mv_line_hue")
            group = st.sidebar.selectbox("Line group (optional)", col_with_none, key="mv_line_grp")
            fig = plotter.line_plot(col_x, col_y, color=hue, line_group=group)
            st.plotly_chart(fig, use_container_width=True)
            add_plotly_download(fig, "lineplot")


render()
