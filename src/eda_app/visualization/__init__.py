"""Plotting classes and utilities for the EDA app."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


class EDAPlotter:
    """Encapsulates all chart-generation methods for a DataFrame.

    Provides both Plotly (interactive) and Seaborn (static) chart types
    for univariate and multivariate analysis.

    Attributes:
        df: The source DataFrame.
        columns: All column names.
        num_vars: Numeric column names.
        cat_vars: Categorical column names.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe
        self.columns = dataframe.columns
        self.num_vars = dataframe.select_dtypes(include="number").columns
        self.cat_vars = dataframe.select_dtypes(include=["object", "category"]).columns

    # --- Plotly charts (interactive) ---

    def box_plot(
        self,
        y: str,
        x: str | None = None,
        color: str | None = None,
    ) -> Any:
        """Create an interactive boxplot with Plotly.

        Args:
            y: Numeric variable for the y-axis.
            x: Optional categorical variable for the x-axis.
            color: Optional variable to color by.

        Returns:
            Plotly Figure object.
        """
        return px.box(
            self.df,
            x=x,
            y=y,
            color=color,
            template="plotly_white",
        )

    def histogram(
        self,
        column: str,
        color: str | None = None,
        nbins: int | None = None,
        range_values: tuple[float, float] | None = None,
    ) -> Any:
        """Create an interactive histogram with marginal violin.

        Args:
            column: Numeric variable to plot.
            color: Optional variable to color by.
            nbins: Number of bins.
            range_values: Tuple of (min, max) to filter data.

        Returns:
            Plotly Figure object.
        """
        data = self.df
        if range_values is not None:
            data = data[data[column].between(*range_values)]
        return px.histogram(
            data,
            x=column,
            nbins=nbins,
            color=color,
            marginal="violin",
            template="plotly_white",
        )

    def scatter_plot(
        self,
        x: str,
        y: str,
        color: str | None = None,
        size: str | None = None,
    ) -> Any:
        """Create an interactive scatter plot.

        Args:
            x: Variable for x-axis.
            y: Variable for y-axis.
            color: Optional variable to color by.
            size: Optional variable to size points by.

        Returns:
            Plotly Figure object.
        """
        return px.scatter(
            self.df,
            x=x,
            y=y,
            color=color,
            size=size,
            template="plotly_white",
        )

    def bar_plot(
        self,
        x: str,
        y: str,
        color: str | None = None,
    ) -> Any:
        """Create an interactive bar plot.

        Args:
            x: Categorical variable for x-axis.
            y: Numeric variable for y-axis.
            color: Optional variable to color by.

        Returns:
            Plotly Figure object.
        """
        return px.bar(
            self.df,
            x=x,
            y=y,
            color=color,
            template="plotly_white",
        )

    def line_plot(
        self,
        x: str,
        y: str,
        color: str | None = None,
        line_group: str | None = None,
    ) -> Any:
        """Create an interactive line plot.

        Args:
            x: Variable for x-axis.
            y: Variable for y-axis.
            color: Optional variable to color lines by.
            line_group: Optional variable to group lines by.

        Returns:
            Plotly Figure object.
        """
        return px.line(
            self.df,
            x=x,
            y=y,
            color=color,
            line_group=line_group,
            template="plotly_white",
        )

    # --- Seaborn/Matplotlib charts (static) ---

    def violin_plot(
        self,
        y: str,
        x: str | None = None,
        hue: str | None = None,
        split: bool = False,
    ) -> plt.Figure:
        """Create a violin plot with Seaborn.

        Args:
            y: Numeric variable for the y-axis.
            x: Optional categorical variable for the x-axis.
            hue: Optional variable to split by color.
            split: Whether to split violins when hue is set.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots()
        kwargs: dict[str, Any] = {
            "data": self.df,
            "x": x,
            "y": y,
            "split": split,
            "ax": ax,
        }
        if hue:
            kwargs["hue"] = hue
            kwargs["palette"] = "husl"
        sns.violinplot(**kwargs)
        return fig

    def swarm_plot(
        self,
        y: str,
        x: str | None = None,
        hue: str | None = None,
        dodge: bool = False,
    ) -> plt.Figure:
        """Create a swarm plot with Seaborn.

        Args:
            y: Numeric variable for the y-axis.
            x: Optional categorical variable for the x-axis.
            hue: Optional variable to split by color.
            dodge: Whether to dodge points when hue is set.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots()
        kwargs: dict[str, Any] = {
            "data": self.df,
            "x": x,
            "y": y,
            "dodge": dodge,
            "ax": ax,
        }
        if hue:
            kwargs["hue"] = hue
            kwargs["palette"] = "husl"
        sns.swarmplot(**kwargs)
        return fig

    def count_plot(
        self,
        column: str,
        hue: str | None = None,
    ) -> plt.Figure:
        """Create a count plot with Seaborn.

        Args:
            column: Categorical variable to count.
            hue: Optional variable to split bars by.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots()
        kwargs: dict[str, Any] = {
            "data": self.df,
            "x": column,
            "ax": ax,
        }
        if hue:
            kwargs["hue"] = hue
            kwargs["palette"] = "pastel"
        chart = sns.countplot(**kwargs)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        return fig

    def dist_plot(self, column: str) -> plt.Figure:
        """Create a distribution plot (histogram + KDE + rug).

        Args:
            column: Numeric variable to plot.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots()
        sns.histplot(self.df[column], kde=True, color="c", ax=ax)
        sns.rugplot(self.df[column], color="c", ax=ax)
        return fig

    def correlation_heatmap(
        self,
        columns: list[str] | None = None,
        method: str = "pearson",
    ) -> plt.Figure:
        """Create a correlation heatmap.

        Args:
            columns: Specific columns to include. If None, uses all numeric.
            method: Correlation method ('pearson', 'kendall', 'spearman').

        Returns:
            Matplotlib Figure object.
        """
        data = (
            self.df[columns].select_dtypes("number") if columns else self.df.select_dtypes("number")
        )
        corr = data.corr(method=method)

        fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.8), max(6, len(corr) * 0.6)))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cmap="RdBu_r",
            center=0,
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        return fig

    def heatmap_pivot(
        self,
        index: str,
        columns: str,
        values: str,
        aggfunc: str | Callable = "mean",
    ) -> plt.Figure:
        """Create a heatmap from a pivot table.

        Args:
            index: Column for pivot table rows.
            columns: Column for pivot table columns.
            values: Column for pivot table values.
            aggfunc: Aggregation function.

        Returns:
            Matplotlib Figure object.
        """
        pivot = self.df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=0,
        ).dropna(axis=1)

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8), max(6, len(pivot) * 0.5)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        return fig
