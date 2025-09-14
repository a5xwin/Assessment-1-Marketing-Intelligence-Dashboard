# Marketing Intelligence Dashboard

A comprehensive Streamlit-based dashboard for analyzing marketing performance and its direct connection to business outcomes, using data from platforms like Facebook, Google, and TikTok.

## Features

- **Advanced KPIs**:  
  - Calculates and monitors ROAS (Return on Ad Spend).  
  - Tracks CAC (Customer Acquisition Cost).  
  - Monitors CTR (Click-Through Rate).  
  - Evaluates MER (Marketing Efficiency Ratio).  

- **Executive Summary**:  
  - Provides a high-level overview of total spend.  
  - Summarizes revenue and profit.  
  - Highlights overall marketing effectiveness.  

- **Channel Deep Dive**:  
  - **Performance Matrix**:  
    - Bubble chart visualizing ROAS vs. CTR.  
    - Bubble size represents total spend.  
  - **Efficiency Radar Chart**:  
    - Compares platforms across multiple KPIs.  
    - Normalized values for apples-to-apples comparison.  

- **Geographic Analysis**:  
  - US choropleth map visualization.  
  - Shows ROAS by state.  
  - Helps identify high-performing regions.  

- **Automated Insights**:  
  - "Intelligent Analytics" tab generates rule-based recommendations.  
  - Provides performance alerts (e.g., "Low ROAS").  
  - Suggests budget reallocation and optimization strategies.  

- **Data Quality Monitoring**:  
  - Dedicated tab for assessing data completeness.  
  - Monitors data freshness.  
  - Ensures consistency for trustworthy insights.  

- **Data Export**:  
  - Download filtered data as CSV files.  
  - Export KPI summaries.  
  - Export top-campaign lists.  

- **Custom Branding**:  
  - Supports custom platform logo integration.  
  - Provides a branded and professional dashboard look.  


## Stack Used

This project is built on a robust Python toolkit. Pandas serves as the backbone for data manipulation and analysis, while NumPy provides essential support for numerical computing. For data visualization, the dashboard leverages Plotly to create rich, interactive charts and relies on Matplotlib for foundational plotting and color gradient generation. SciPy is included for advanced statistical analysis. The entire user interface is built and served using Streamlit, an open-source framework for creating and sharing web apps for data science.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## App Working Live Demo
   - Visit: https://assessment-1-marketing-intelligence-dashboard-ash.streamlit.app/


## Installing Locally

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

4. **Running the application**
   ```bash
   streamlit run app.py
   ```


