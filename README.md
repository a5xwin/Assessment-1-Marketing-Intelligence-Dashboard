Marketing Intelligence Dashboard

This project is an interactive Business Intelligence (BI) dashboard built with Streamlit. It is designed to help a business stakeholder analyze marketing performance and its direct connection to business outcomes. The dashboard integrates campaign-level data from Facebook, Google, and TikTok with daily business performance data. 

The primary goal is to move beyond surface-level reporting by providing actionable insights, advanced visualizations, and context-aware recommendations that inform strategic decisions. 

Key Features

The dashboard is organized into a multi-tab interface for a clear and structured user experience.

    Core Functionality

        Unified Data Model: Integrates four distinct datasets into a single, analysis-ready source. 

    Derived Metrics Engine: Automatically calculates essential KPIs on-the-fly, including Return on Ad Spend (ROAS), Customer Acquisition Cost (CAC), Click-Through Rate (CTR), Cost Per Click (CPC), and profit margins.

    Dynamic Filtering: All analytics can be filtered by a specific date range and by advertising platform, allowing for granular analysis.

Tab 1: Executive Summary

    Provides a high-level overview of business health and marketing efficiency.

    Features KPI cards for key metrics like Total Spend, Total Revenue, Gross Profit, overall ROAS, and overall CAC.

    Includes a time-series chart visualizing the trends of revenue, profit, and ad spend.

Tab 2: Channel Analysis

    Delivers a comparative analysis of performance across Facebook, Google, and TikTok.

    Unique Feature: Performance Matrix. An interactive bubble chart that plots platforms based on ROAS (x-axis) and CTR (y-axis). The size of each bubble represents the total spend, providing a multi-dimensional view of channel efficiency.

    Unique Feature: Efficiency Radar Chart. Normalizes disparate KPIs (ROAS, CTR, CPC, CAC) onto a single radar chart to provide a true "apples-to-apples" comparison of each platform's strengths and weaknesses.

Tab 3: Strategy & Geographic Analysis

    Allows for a deep dive into the performance of individual campaigns and tactics.

    Presents a detailed, sortable data table for granular campaign-level analysis.

    Unique Feature: Geographic Performance Analysis. Visualizes where campaigns are most effective through:

        A US Choropleth Map colored by state-level ROAS.

        A horizontal bar chart of the top 10 states by total ad spend.

Tab 4: Business Impact Analysis

    Focuses directly on the assessment's core task: connecting marketing activity to business outcomes. 

        Unique Feature: Correlation Heatmap. A matrix that visualizes the statistical correlation between marketing inputs (e.g., spend, clicks) and business outputs (e.g., orders, revenue, new customers), revealing which marketing actions most strongly influence the bottom line.

        Includes scatter plots with OLS trendlines and R² values to quantify the relationship between daily ad spend and key outcomes like orders and new customer acquisition.

    Tab 5: Performance Highlights & Intelligent Recommendations

        Moves from analysis to action by automatically surfacing key takeaways.

        Unique Feature: Automated Insights Engine. Generates context-aware, rule-based text outputs based on the filtered data. It provides strategic recommendations (e.g., "Consider reallocating budget from Platform X to Platform Y") and performance alerts (e.g., "Low ROAS Alert: Overall ROAS is below the profitability threshold").

        Includes a "Campaign Spotlight" that automatically identifies and displays the top 10 performing campaigns by ROAS.

    Tab 6: Data Quality & Exports

        A unique section focused on building trust in the data and providing utility.

        Unique Feature: Data Quality Monitor. Assesses the filtered data for completeness, freshness, and consistency. It generates an overall quality score and flags specific issues like negative spend, impossible CTR values, or stale data.

        Data Export Center: Allows the user to download a KPI summary, the full filtered dataset, or a list of top-performing campaigns as a CSV file.

Technical Implementation

    Framework: Streamlit

    Core Libraries: Pandas for data manipulation, Plotly for advanced interactive visualizations.

    Data Pipeline: The data preparation process involves:

        Loading four separate CSV files.

        Cleaning and standardizing column names.

        Appending a platform identifier to each marketing dataset.

        Merging marketing and business dataframes on the date column.

        Calculating a suite of derived metrics for analysis.

    Styling: The dashboard uses custom CSS to inject unique fonts and create a polished, professional layout, demonstrating attention to the final product's usability.

How to Run Locally

    Prerequisites: Python 3.8+ installed.

    Clone Repository:
    Bash

git clone <your-repo-url>
cd <your-repo-name>

Install Dependencies:
Bash

pip install -r requirements.txt

Directory Structure:

    Ensure the CSV files are in a /data subdirectory.

    Ensure the font files (thin.ttf, bold.ttf) are in the root directory.

Run the App:
Bash

streamlit run app.py
