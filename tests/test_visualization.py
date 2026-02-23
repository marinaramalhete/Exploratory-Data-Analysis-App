"""Smoke tests for visualization — ensure charts are generated without errors."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.eda_app.visualization import EDAPlotter


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for chart testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 50, 22, 28, 33],
            "salary": [50000, 60000, 70000, 80000, 90000, 45000, 55000, 65000],
            "dept": ["Eng", "Sales", "Eng", "Sales", "Eng", "Sales", "Eng", "Sales"],
            "city": ["NYC", "LA", "NYC", "LA", "NYC", "LA", "NYC", "LA"],
        }
    )


@pytest.fixture
def plotter(sample_df: pd.DataFrame) -> EDAPlotter:
    """Create an EDAPlotter instance."""
    return EDAPlotter(sample_df)


class TestPlotlyCharts:
    """Smoke tests for Plotly chart methods."""

    def test_box_plot(self, plotter: EDAPlotter) -> None:
        fig = plotter.box_plot("age")
        assert fig is not None

    def test_box_plot_with_options(self, plotter: EDAPlotter) -> None:
        fig = plotter.box_plot("age", x="dept", color="city")
        assert fig is not None

    def test_histogram(self, plotter: EDAPlotter) -> None:
        fig = plotter.histogram("age", nbins=10, range_values=(20.0, 55.0))
        assert fig is not None

    def test_scatter_plot(self, plotter: EDAPlotter) -> None:
        fig = plotter.scatter_plot("age", "salary")
        assert fig is not None

    def test_bar_plot(self, plotter: EDAPlotter) -> None:
        fig = plotter.bar_plot("dept", "salary")
        assert fig is not None

    def test_line_plot(self, plotter: EDAPlotter) -> None:
        fig = plotter.line_plot("age", "salary")
        assert fig is not None


class TestSeabornCharts:
    """Smoke tests for Seaborn/Matplotlib chart methods."""

    def test_violin_plot(self, plotter: EDAPlotter) -> None:
        fig = plotter.violin_plot("age", x="dept")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_swarm_plot(self, plotter: EDAPlotter) -> None:
        fig = plotter.swarm_plot("age", x="dept")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_count_plot(self, plotter: EDAPlotter) -> None:
        fig = plotter.count_plot("dept")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dist_plot(self, plotter: EDAPlotter) -> None:
        fig = plotter.dist_plot("age")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correlation_heatmap(self, plotter: EDAPlotter) -> None:
        fig = plotter.correlation_heatmap()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correlation_with_method(self, plotter: EDAPlotter) -> None:
        fig = plotter.correlation_heatmap(method="spearman")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_pivot(self, plotter: EDAPlotter) -> None:
        fig = plotter.heatmap_pivot("dept", "city", "salary", aggfunc="mean")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestEdgeCases:
    """Edge-case tests for chart methods."""

    def test_empty_dataframe_heatmap(self) -> None:
        df = pd.DataFrame(columns=["age", "salary", "dept"])
        plotter = EDAPlotter(df)
        try:
            fig = plotter.correlation_heatmap()
        except (ValueError, TypeError):
            pass  # empty DF can raise — acceptable
        else:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_single_row_heatmap(self) -> None:
        df = pd.DataFrame([{"age": 30, "salary": 100_000, "dept": "Eng"}])
        plotter = EDAPlotter(df)
        fig = plotter.correlation_heatmap()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_mixed_columns_filters_numeric(self) -> None:
        df = pd.DataFrame(
            {
                "age": [25, 35, 45],
                "salary": [70_000, 90_000, 110_000],
                "dept": ["sales", "eng", "hr"],
            }
        )
        plotter = EDAPlotter(df)
        fig = plotter.correlation_heatmap(columns=["age", "salary", "dept"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_no_numeric_columns(self) -> None:
        df = pd.DataFrame({"dept": ["sales", "eng", "hr"], "city": ["NY", "SF", "LA"]})
        plotter = EDAPlotter(df)
        try:
            fig = plotter.correlation_heatmap()
        except (ValueError, TypeError):
            pass  # controlled exception is acceptable
        else:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
