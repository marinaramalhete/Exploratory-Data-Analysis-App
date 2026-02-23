# 🔍 Exploratory Data Analysis App

[![CI](https://github.com/marinaramalhete/Exploratory-Data-Analysis-App/actions/workflows/ci.yml/badge.svg)](https://github.com/marinaramalhete/Exploratory-Data-Analysis-App/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)

An interactive web application for **Exploratory Data Analysis** — upload your dataset (CSV, Excel, or Parquet) and get comprehensive statistics, interactive visualizations, and an automated profiling report.

## Features

- **Multi-format support** — CSV, Excel (.xlsx), and Parquet files
- **Overview** — Summary statistics, data types, and missing value analysis
- **Univariate analysis** — Histograms, boxplots, distribution plots with descriptive stats and quantiles
- **Multivariate analysis** — 10 chart types: correlation heatmap, scatter, violin, swarm, bar, line, count, pivot heatmap, and more
- **Auto-profiling** — Comprehensive automated report with outlier detection (IQR + Z-score), data quality alerts, distribution analysis, and correlation matrix
- **Chart export** — Download any chart as PNG
- **Data export** — Download statistics and processed data as CSV
- **Interactive** — Plotly for interactive charts + Seaborn for statistical plots

## Architecture

```
src/eda_app/
├── Home.py                 # Entry point + Home page
├── data/
│   ├── __init__.py         # Data loading (CSV/Excel/Parquet)
│   └── preprocessing.py    # Missing value imputation
├── stats/
│   ├── __init__.py         # Descriptive stats, quantiles, summaries
│   └── profiling.py        # Automated profiling report
├── visualization/
│   └── __init__.py         # EDAPlotter class (Plotly + Seaborn)
├── components/
│   └── download.py         # Chart/data export utilities
└── pages/
    ├── 1_Overview.py
    ├── 2_Univariate.py
    ├── 3_Multivariate.py
    └── 4_Profiling.py
tests/
├── test_loader.py
├── test_stats.py
├── test_preprocessing.py
└── test_visualization.py
```

## Getting Started

### Prerequisites

- Python 3.11+
- [conda](https://docs.conda.io/) (recommended) or pip

### Installation

```bash
# Clone the repo
git clone https://github.com/marinaramalhete/Exploratory-Data-Analysis-App.git
cd Exploratory-Data-Analysis-App

# Create environment
conda create -n eda-app python=3.11 -y
conda activate eda-app

# Install dependencies
pip install -r requirements.txt
```

### Running the app

```bash
streamlit run src/eda_app/Home.py
```

### Running tests

```bash
pip install pytest
pytest tests/ -v
```

## Development

```bash
# Install dev dependencies
pip install ruff mypy pytest pre-commit pandas-stubs

# Linting
ruff check src/ pages/ tests/
ruff format src/ pages/ tests/

# Pre-commit hooks
pre-commit install
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Framework | Streamlit 1.40+ |
| Data | Pandas, NumPy, SciPy |
| Visualization | Plotly, Seaborn, Matplotlib |
| Testing | Pytest |
| Linting | Ruff |
| CI/CD | GitHub Actions |

## Author

**Marina Ramalhete Masid** — [LinkedIn](https://www.linkedin.com/in/marinaramalhete/) · [GitHub](https://github.com/marinaramalhete)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
