# Marketing Intelligence Dashboard:-

A comprehensive Streamlit-based dashboard for analyzing marketing performance and its direct connection to business outcomes, using data from platforms like Facebook, Google, and TikTok.

## Features:-

- **Advanced KPIs**: Calculates and monitors key metrics including ROAS (Return on Ad Spend), CAC (Customer Acquisition Cost), CTR (Click-Through Rate), and MER (Marketing Efficiency Ratio).
- **Executive Summary**: A high-level overview of total spend, revenue, profit, and overall marketing effectiveness.
- **Channel Deep Dive:**: Performance Matrix: A unique bubble chart visualizing ROAS vs. CTR, with bubble size representing total spend.
Efficiency Radar Chart: Compares platforms across multiple normalized KPIs for a true apples-to-apples performance view.
- **Geographic Analysis**: A US choropleth map visualizes ROAS by state to identify high-performing regions.
- **Automated Insights**: An "Intelligent Analytics" tab that generates rule-based recommendations and performance alerts (e.g., "Low ROAS," "Budget reallocation suggestions").
- **Data Quality Monitoring**: A dedicated tab that assesses the underlying data for completeness, freshness, and consistency, building trust in the insights.
- **Data Export**: Allows users to download filtered data, KPI summaries, and top-campaign lists as CSV files.

## Stack Used:-

This project is built on a robust Python toolkit. Pandas serves as the backbone for data manipulation and analysis, while NumPy provides essential support for numerical computing. For data visualization, the dashboard leverages Plotly to create rich, interactive charts and relies on Matplotlib for foundational plotting and color gradient generation. SciPy is included for advanced statistical analysis. The entire user interface is built and served using Streamlit, an open-source framework for creating and sharing web apps for data science.

## Prerequisites:-

- Python 3.8 or higher
- pip package manager

## Installation:-

1. **Clone the repository**
   ```bash
   https://github.com/a5xwin/Assessment-1-Marketing-Intelligence-Dashboard
   cd Assessment-1-Marketing-Intelligence-Dashboard-main
   ```

2. **Install required packages**
   ```bash
   pip install streamlit pandas plotly numpy
   ```

3. **Getting Your Data Ready**
   - Ensure you have a data folder in the root directory.
   - Place the following four CSV files inside the data folder:
     - `Facebook.csv` - Facebook campaign-level marketing data
     - `Google.csv` - Google campaign-level marketing data
     - `TikTok.csv` - TikTok campaign-level marketing data
     - `business.csv` - Daily business performance data

## Testing the Application:-
   - Visit: https://assessment-1-marketing-intelligence-dashboard-ash.streamlit.app/

## Running the Application Locally

```bash
streamlit run app.py
```

