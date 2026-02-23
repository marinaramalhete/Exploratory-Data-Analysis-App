"""Automated data profiling — generates a comprehensive EDA report."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from src.eda_app.data import get_categorical_columns, get_numeric_columns


def _detect_outliers_iqr(series: pd.Series) -> int:
    """Count outliers using the IQR method (1.5×IQR rule).

    Args:
        series: Numeric series to check.

    Returns:
        Number of outliers.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def _detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> int:
    """Count outliers using the Z-score method.

    Args:
        series: Numeric series to check.
        threshold: Z-score threshold (default 3.0).

    Returns:
        Number of outliers.
    """
    if series.std() == 0:
        return 0
    z_scores = np.abs((series - series.mean()) / series.std())
    return int((z_scores > threshold).sum())


def generate_alerts(df: pd.DataFrame) -> list[dict[str, str]]:
    """Generate data quality alerts for the dataset.

    Checks for: high missing rates, constant columns, high cardinality,
    high correlation between features, and outliers.

    Args:
        df: Input DataFrame.

    Returns:
        List of alert dicts with 'type' (warning/info/error), 'column', 'message'.
    """
    alerts: list[dict[str, str]] = []
    n_rows = len(df)

    for col in df.columns:
        missing_pct = df[col].isna().sum() / n_rows * 100

        # High missing
        if missing_pct > 50:
            alerts.append(
                {
                    "type": "🔴",
                    "column": col,
                    "message": f"{missing_pct:.1f}% missing values",
                }
            )
        elif missing_pct > 15:
            alerts.append(
                {
                    "type": "🟡",
                    "column": col,
                    "message": f"{missing_pct:.1f}% missing values",
                }
            )

        # Constant column
        if df[col].nunique() <= 1:
            alerts.append(
                {
                    "type": "🔴",
                    "column": col,
                    "message": "Constant column (only 1 unique value)",
                }
            )

        # High cardinality
        if df[col].dtype == "object" and df[col].nunique() > n_rows * 0.8:
            alerts.append(
                {
                    "type": "🟡",
                    "column": col,
                    "message": f"High cardinality: {df[col].nunique()} unique values ({df[col].nunique() / n_rows * 100:.0f}%)",
                }
            )

    # High correlation
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
        for col in upper_tri.columns:
            high_corr = upper_tri[col][upper_tri[col] > 0.95]
            for idx_col, val in high_corr.items():
                alerts.append(
                    {
                        "type": "🟡",
                        "column": f"{idx_col} ↔ {col}",
                        "message": f"High correlation: {val:.3f}",
                    }
                )

    # Outliers
    for col in numeric_cols:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue
        n_outliers = _detect_outliers_iqr(clean)
        if n_outliers > n_rows * 0.05:
            alerts.append(
                {
                    "type": "🟡",
                    "column": col,
                    "message": f"{n_outliers} outliers detected ({n_outliers / n_rows * 100:.1f}%) via IQR method",
                }
            )

    return alerts


def render_profiling(df: pd.DataFrame) -> None:
    """Render the full auto-profiling report in Streamlit.

    Sections: Alerts, Overview, Numeric distributions, Categorical distributions,
    Correlation matrix, Missing values heatmap, Outlier summary.

    Args:
        df: The DataFrame to profile.
    """
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    # --- Alerts ---
    alerts = generate_alerts(df)
    if alerts:
        st.subheader("⚠️ Data Quality Alerts")
        alert_df = pd.DataFrame(alerts)
        st.dataframe(
            alert_df.rename(columns={"type": "Severity", "column": "Column", "message": "Alert"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No data quality issues detected!")

    # --- Overview ---
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Numeric", f"{len(numeric_cols)}")
    col4.metric("Categorical", f"{len(categorical_cols)}")

    col5, col6, col7 = st.columns(3)
    total_missing = df.isna().sum().sum()
    total_cells = df.size
    col5.metric("Missing Cells", f"{total_missing:,}")
    col6.metric("Missing %", f"{total_missing / total_cells * 100:.2f}%")
    col7.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    with st.expander("📊 Data Types", expanded=False):
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str).values,
                    "Missing": df.isna().sum().values,
                    "Missing %": (df.isna().sum() / len(df) * 100).round(2).values,
                    "Unique": df.nunique().values,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # --- Numeric distributions ---
    if numeric_cols:
        st.subheader("📈 Numeric Variable Distributions")
        for col in numeric_cols:
            with st.expander(f"**{col}**", expanded=False):
                stats_col, chart_col = st.columns([1, 2])

                with stats_col:
                    desc = df[col].describe()
                    st.dataframe(desc.to_frame().T, use_container_width=True)

                    n_outliers_iqr = _detect_outliers_iqr(df[col].dropna())
                    n_outliers_z = _detect_outliers_zscore(df[col].dropna())
                    st.caption(f"Outliers — IQR: {n_outliers_iqr} | Z-score: {n_outliers_z}")

                with chart_col:
                    fig = px.histogram(
                        df,
                        x=col,
                        marginal="box",
                        template="plotly_white",
                        title=f"Distribution of {col}",
                    )
                    fig.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    # --- Categorical distributions ---
    if categorical_cols:
        st.subheader("📊 Categorical Variable Distributions")
        for col in categorical_cols:
            with st.expander(f"**{col}** ({df[col].nunique()} unique)", expanded=False):
                value_counts = df[col].value_counts()

                stats_col, chart_col = st.columns([1, 2])

                with stats_col:
                    desc = df[col].describe().to_frame().T
                    st.dataframe(desc, use_container_width=True)

                with chart_col:
                    # Show top 20 categories max
                    top_n = value_counts.head(20)
                    fig = px.bar(
                        x=top_n.values,
                        y=top_n.index,
                        orientation="h",
                        template="plotly_white",
                        title=f"Top categories in {col}",
                        labels={"x": "Count", "y": col},
                    )
                    fig.update_layout(height=max(250, len(top_n) * 25), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    # --- Correlation matrix ---
    if len(numeric_cols) >= 2:
        st.subheader("🔗 Correlation Matrix")
        method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"],
            key="profiling_corr_method",
        )
        corr = df[numeric_cols].corr(method=method)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            template="plotly_white",
            title=f"{method.capitalize()} Correlation Matrix",
        )
        fig.update_layout(height=max(400, len(numeric_cols) * 40))
        st.plotly_chart(fig, use_container_width=True)

    # --- Missing values heatmap ---
    if df.isna().any().any():
        st.subheader("🕳️ Missing Values Pattern")
        fig, ax = plt.subplots(figsize=(10, max(3, len(df.columns) * 0.3)))
        sns.heatmap(
            df.isna().T,
            cbar=True,
            yticklabels=True,
            cmap="YlOrRd",
            ax=ax,
        )
        ax.set_title("Missing Values Heatmap (yellow = missing)")
        ax.set_xlabel("Row Index")
        st.pyplot(fig)
        plt.close(fig)

    # --- Sample data ---
    st.subheader("🔍 Data Sample")
    sample_size = min(10, len(df))
    st.dataframe(df.sample(sample_size, random_state=42), use_container_width=True)
