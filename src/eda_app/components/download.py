"""Chart export utilities — download Plotly and Matplotlib figures."""

from __future__ import annotations

import io
from typing import Any

import streamlit as st


def add_plotly_download(fig: Any, filename: str = "chart") -> None:
    """Add a download button for a Plotly figure as PNG.

    Args:
        fig: Plotly Figure object.
        filename: Base filename (without extension).
    """
    try:
        img_bytes = fig.to_image(format="png", scale=2)
        st.download_button(
            label="📥 Download PNG",
            data=img_bytes,
            file_name=f"{filename}.png",
            mime="image/png",
            key=f"dl_plotly_{filename}_{id(fig)}",
        )
    except Exception:
        st.caption("PNG export not available (kaleido may not be installed).")


def add_matplotlib_download(fig: Any, filename: str = "chart") -> None:
    """Add a download button for a Matplotlib figure as PNG.

    Args:
        fig: Matplotlib Figure object.
        filename: Base filename (without extension).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="📥 Download PNG",
        data=buf,
        file_name=f"{filename}.png",
        mime="image/png",
        key=f"dl_mpl_{filename}_{id(fig)}",
    )


def add_csv_download(df: Any, filename: str = "data") -> None:
    """Add a download button for a DataFrame as CSV.

    Args:
        df: Pandas DataFrame.
        filename: Base filename (without extension).
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv",
        key=f"dl_csv_{filename}_{id(df)}",
    )
