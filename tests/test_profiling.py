"""Tests for automated profiling alerts."""

from __future__ import annotations

import pandas as pd

from src.eda_app.stats.profiling import generate_alerts


class TestGenerateAlerts:
    """Tests for generate_alerts covering all alert branches."""

    def test_high_missingness_alert(self) -> None:
        """Columns above 15% and 50% missing should trigger alerts."""
        df = pd.DataFrame(
            {
                "low_missing": list(range(90)) + [None] * 10,  # 10% — no missing alert
                "moderate_missing": list(range(80)) + [None] * 20,  # 20% — warning
                "high_missing": list(range(40)) + [None] * 60,  # 60% — critical
            }
        )
        alerts = generate_alerts(df)
        missing_alerts = [a for a in alerts if "missing" in a["message"].lower()]
        columns = [a["column"] for a in missing_alerts]

        assert "moderate_missing" in columns
        assert "high_missing" in columns
        assert "low_missing" not in columns

    def test_constant_column_alert(self) -> None:
        """A column with a single unique value should trigger an alert."""
        df = pd.DataFrame(
            {
                "constant": [42] * 20,
                "varied": list(range(20)),
            }
        )
        alerts = generate_alerts(df)
        columns = [a["column"] for a in alerts]

        assert "constant" in columns

    def test_high_cardinality_alert(self) -> None:
        """A categorical column where every value is unique should trigger an alert."""
        df = pd.DataFrame(
            {
                "unique_ids": [f"id_{i}" for i in range(50)],
                "category": ["A", "B"] * 25,
            }
        )
        alerts = generate_alerts(df)
        columns = [a["column"] for a in alerts]

        assert "unique_ids" in columns
        assert "category" not in columns

    def test_high_correlation_alert(self) -> None:
        """Two perfectly correlated columns should trigger a correlation alert."""
        base = list(range(50))
        df = pd.DataFrame(
            {
                "x": base,
                "y": [v * 2 for v in base],
                "z": list(range(50, 100)),
            }
        )
        alerts = generate_alerts(df)
        joined = " ".join(a["column"] for a in alerts)

        assert "x" in joined and "y" in joined

    def test_outlier_alert(self) -> None:
        """A column with obvious outliers should trigger an outlier alert."""
        df = pd.DataFrame(
            {
                "values": [10.0] * 40 + [1000.0, 1200.0, 1500.0],
            }
        )
        alerts = generate_alerts(df)
        columns = [a["column"] for a in alerts]

        assert "values" in columns

    def test_no_alerts_for_clean_data(self) -> None:
        """A well-behaved dataset should produce no alerts."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cat": ["x", "y", "z", "x", "y"],
            }
        )
        alerts = generate_alerts(df)

        assert len(alerts) == 0
