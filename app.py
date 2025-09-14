import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import ttest_ind
import base64
import json
from datetime import datetime

@st.cache_data
def inject_custom_fonts():
    def get_font_base64(font_path):
        with open(font_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    try:
        regular_font_base64 = get_font_base64('thin.ttf')
        bold_font_base64 = get_font_base64('bold.ttf')
    except FileNotFoundError as e:
        st.error(f"Font file not found. Please ensure 'thin.ttf' and 'bold.ttf' are in the root directory. Error: {e}")
        return

    font_css = f"""
    <style>
    /* Define the 'customthin' font */
    @font-face {{
        font-family: 'customthin';
        src: url(data:font/ttf;base64,{regular_font_base64}) format('truetype');
        font-weight: normal;
        font-style: normal;
    }}

    /* Define the 'custombold' font */
    @font-face {{
        font-family: 'custombold';
        src: url(data:font/ttf;base64,{bold_font_base64}) format('truetype');
        font-weight: normal;
        font-style: normal;
    }}

    /* Apply 'customthin' to the main content container */
    div[data-testid="stAppViewContainer"] {{
        font-family: 'customthin', sans-serif;
    }}

    /* Apply 'custombold' to all heading elements */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'custombold', sans-serif !important;
    }}
    
    /* Increase tab title sizes */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 16px !important;
        font-weight: 600 !important;
    }}
    
    .stTabs [data-baseweb="tab-list"] button {{
        font-size: 16px !important;
        padding: 12px 20px !important;  
    }}

    div[data-testid="stAppViewContainer"] h1 {{
        margin-top: -125px !important;
        margin-left: 150px !important;
    }}

    </style>
    """
    st.markdown(font_css, unsafe_allow_html=True)
inject_custom_fonts()



# browser tab
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="〽", 
    layout="wide",
)

st.markdown(
    """
    <style>
        .top-right-icon {
            position: relative;
            top: -10px;
            left: 10px; #move towards left
            z-index: 9999;
        }
        .top-right-icon img {
            width: 120px; /* Adjust size here */
            height: 120px;
            opacity: 0.8; /* Optional: for a softer look */
        }
    </style>
    <div class="top-right-icon">
        <img src="https://cdn-icons-png.flaticon.com/512/6821/6821002.png">
    </div>
    """,
    unsafe_allow_html=True,
)

# --- DATA LOADING AND PREPARATION ---
@st.cache_data
def load_and_prepare_data():
    try:
        facebook_df = pd.read_csv('data/Facebook.csv')
        google_df = pd.read_csv('data/Google.csv')
        tiktok_df = pd.read_csv('data/TikTok.csv')
        business_df = pd.read_csv('data/business.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Make sure the CSV files are in the 'data' folder.")
        return None

    def clean_col_names(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('#_of_', 'num_')
        return df

    facebook_df = clean_col_names(facebook_df)
    google_df = clean_col_names(google_df)
    tiktok_df = clean_col_names(tiktok_df)
    business_df = clean_col_names(business_df)
    
    facebook_df['platform'] = 'Facebook'
    google_df['platform'] = 'Google'
    tiktok_df['platform'] = 'TikTok'
    marketing_df = pd.concat([facebook_df, google_df, tiktok_df], ignore_index=True)

    marketing_df['date'] = pd.to_datetime(marketing_df['date'])
    business_df['date'] = pd.to_datetime(business_df['date'])

    df = pd.merge(marketing_df, business_df, on='date', how='left')
    df.fillna(0, inplace=True)

    # Calculated metrics
    df['ctr'] = np.where(df['impression'] > 0, (df['clicks'] / df['impression']) * 100, 0)
    df['cpc'] = np.where(df['clicks'] > 0, df['spend'] / df['clicks'], 0)
    df['roas'] = np.where(df['spend'] > 0, df['attributed_revenue'] / df['spend'], 0)
    df['mer'] = np.where(df['spend'] > 0, df['total_revenue'] / df['spend'], 0)
    df['cac'] = np.where(df['new_customers'] > 0, df['spend'] / df['new_customers'], 0)
    df['gross_margin'] = np.where(df['total_revenue'] > 0, (df['gross_profit'] / df['total_revenue']) * 100, 0)

    return df

def display_top_campaigns(df):
    """Identifies and displays the top 10 campaigns by ROAS in a styled table."""
    st.subheader("🌟 Campaign Spotlight: Top Performers")
    st.markdown(
        "This table highlights the top 10 most profitable campaigns based on Return on Ad Spend (ROAS) from your selected data.",
        help="ROAS = Attributed Revenue / Spend"
    )

    # Define the columns we want to show
    display_cols = ['platform', 'campaign', 'tactic', 'spend', 'attributed_revenue', 'clicks', 'roas']
    
    # Filter for campaigns with positive ROAS and get the top 10
    top_performers = df[df['roas'] > 0].nlargest(10, 'roas')[display_cols]

    if top_performers.empty:
        st.info("No campaigns with positive ROAS found in the selected data.")
        return

    def assign_performance_tier(roas):
        if roas > 5:
            return "Top Tier"
        elif roas > 3:
            return "Strong"
        else: 
            return "Good"
    
    top_performers['Performance Tier'] = top_performers['roas'].apply(assign_performance_tier)
    
    # Reorder columns for a better presentation, putting the tier first
    top_performers = top_performers[['Performance Tier', 'platform', 'campaign', 'tactic', 'roas', 'spend', 'attributed_revenue', 'clicks']]
    
    # Display the styled dataframe
    st.dataframe(
        top_performers.style.format({
            'spend': '${:,.0f}',
            'attributed_revenue': '${:,.0f}',
            'roas': '{:.2f}x',
            'clicks': '{:,.0f}'
        }).background_gradient(subset=['roas'], cmap='Greens'), 
        use_container_width=True,
        hide_index=True
    )

def show_customer_growth_metrics(filtered_data):
    """Generates and displays customer acquisition KPIs and trend charts."""
    st.subheader("Customer Growth & Acquisition")

    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
        return

    # --- 1. KPI Calculations ---
    # De-duplicate to get accurate daily business-wide counts
    unique_daily_entries = filtered_data[['date', 'new_customers', 'num_orders']].drop_duplicates()
    
    # Calculate overall CAC
    total_spend = filtered_data['spend'].sum()
    total_new_customers = unique_daily_entries['new_customers'].sum()
    avg_acquisition_cost = total_spend / total_new_customers if total_new_customers > 0 else 0
    
    # Calculate daily average new customers
    avg_daily_recruits = unique_daily_entries['new_customers'].mean()

    # Calculate new customer rate
    total_orders = unique_daily_entries['num_orders'].sum()
    first_time_buyer_ratio = (total_new_customers / total_orders * 100) if total_orders > 0 else 0

    # --- 2. Display KPI Cards using st.metric ---
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Customer Acquisition Cost (CAC)", value=f"${avg_acquisition_cost:.2f}", help="Total Spend / New Customers")
    col2.metric(label="New Customers per Day (Avg)", value=f"{avg_daily_recruits:.1f}")
    col3.metric(label="First-Time Buyer Rate", value=f"{first_time_buyer_ratio:.1f}%", help="Percentage of total orders from new customers")
    
    # --- 3. Display Trend Charts ---
    col1, col2 = st.columns(2)
    with col1:
        daily_growth_df = unique_daily_entries.groupby('date')['new_customers'].sum().reset_index()
        growth_trend_fig = px.line(
            daily_growth_df,
            x='date', y='new_customers',
            title='Daily New Customers Acquired',
            template='plotly_white'
        )
        st.plotly_chart(growth_trend_fig, use_container_width=True)
    
    with col2:
        spend_per_day = filtered_data.groupby('date')['spend'].sum()
        new_cust_per_day = unique_daily_entries.set_index('date')['new_customers']
        
        daily_cost_df = pd.merge(spend_per_day, new_cust_per_day, on='date').reset_index()
        daily_cost_df['daily_cac'] = np.where(
            daily_cost_df['new_customers'] > 0,
            daily_cost_df['spend'] / daily_cost_df['new_customers'],
            0
        )
        
        cost_trend_fig = px.line(
            daily_cost_df,
            x='date', y='daily_cac',
            title='Daily Cost per Acquisition (CAC)',
            template='plotly_white'
        )
        st.plotly_chart(cost_trend_fig, use_container_width=True)
        

def display_export_options(df):
    """Creates a section with buttons to download various data reports."""
    st.markdown("---")
    st.subheader("Data Export Center")
    st.markdown("Download the filtered data for offline analysis or reporting.")

    total_spend = df['spend'].sum()
    total_attributed_revenue = df['attributed_revenue'].sum()
    overall_roas = total_attributed_revenue / total_spend if total_spend > 0 else 0
    business_summary = df[['date', 'num_orders', 'new_customers']].drop_duplicates()
    total_orders = business_summary['num_orders'].sum()
    total_new_customers = business_summary['new_customers'].sum()
    overall_cac = total_spend / total_new_customers if total_new_customers > 0 else 0

    summary_data = {
        'Metric': ['Total Spend', 'Attributed Revenue', 'Overall ROAS', 'Total Orders', 'New Customers', 'Overall CAC'],
        'Value': [f'${total_spend:,.2f}', f'${total_attributed_revenue:,.2f}', 
                  f'{overall_roas:.2f}x', f'{total_orders:,.0f}', f'{total_new_customers:,.0f}', f'${overall_cac:.2f}']
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = summary_df.to_csv(index=False).encode('utf-8')

    performance_csv = df.to_csv(index=False).encode('utf-8')

    top_performers_df = df[df['roas'] > 0].nlargest(20, 'roas')
    top_performers_csv = top_performers_df.to_csv(index=False).encode('utf-8')
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
           label="Download KPI Summary",
           data=summary_csv,
           file_name=f"kpi_summary_{datetime.now().strftime('%Y%m%d')}.csv",
           mime="text/csv",
           use_container_width=True
        )

    with col2:
        st.download_button(
           label="Download Full Filtered Data",
           data=performance_csv,
           file_name=f"full_performance_data_{datetime.now().strftime('%Y%m%d')}.csv",
           mime="text/csv",
           use_container_width=True
        )

    with col3:
        st.download_button(
           label="Download Top 20 Campaigns",
           data=top_performers_csv,
           file_name=f"top_20_campaigns_{datetime.now().strftime('%Y%m%d')}.csv",
           mime="text/csv",
           use_container_width=True
        )


def display_geographic_analysis(df):
    """Generates and displays a geographic performance map and bar chart."""
    st.subheader("Geographic Performance Analysis")
    st.markdown("*Visualize where your campaigns are most effective across the US.*")

    # Aggregate core metrics by state. We sum first to calculate accurate rates later.
    state_metrics = df.groupby('state').agg(
        total_spend=('spend', 'sum'),
        total_attributed_revenue=('attributed_revenue', 'sum'),
        total_clicks=('clicks', 'sum'),
        total_impressions=('impression', 'sum')
    ).reset_index()

    # Calculate accurate, weighted averages for ROAS and CTR
    state_metrics['roas'] = np.where(
        state_metrics['total_spend'] > 0,
        state_metrics['total_attributed_revenue'] / state_metrics['total_spend'],
        0
    )
    state_metrics['ctr'] = np.where(
        state_metrics['total_impressions'] > 0,
        (state_metrics['total_clicks'] / state_metrics['total_impressions']) * 100,
        0
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### ROAS by State")
        fig_map = px.choropleth(
            state_metrics,
            locations='state',
            locationmode='USA-states',
            color='roas',
            scope='usa',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            hover_name='state',
            hover_data={ # Customizing hover data for clarity
                'state': False,
                'total_spend': ':.2f',
                'total_attributed_revenue': ':.2f',
                'roas': ':.2f'
            }
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.markdown("##### Top 10 States by Spend")
        top_states = state_metrics.nlargest(10, 'total_spend').sort_values('total_spend', ascending=True)
        fig_bar = px.bar(
            top_states,
            x='total_spend',
            y='state',
            orientation='h',
            color='roas',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            text='total_spend',
            labels={'roas': 'ROAS'}
        )
        fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='inside')
        fig_bar.update_layout(
            yaxis_title=None,
            xaxis_title="Total Spend (USD)",
            margin={"r":20,"t":20,"l":0,"b":0} 
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def display_correlation_heatmap(df):
            st.subheader("Connecting Marketing to Business Outcomes")
            st.markdown("*This heatmap shows how different performance indicators move in relation to one another. A strong positive (dark green) or negative (dark purple) score indicates a powerful relationship.*")

            correlation_df = df[[
                'spend', 'impression', 'clicks', 'num_orders', 'total_revenue', 
                'gross_profit', 'new_customers', 'ctr', 'cpc', 'roas'
            ]].rename(columns={
                'num_orders': 'Orders', 'total_revenue': 'Revenue', 'gross_profit': 'Profit',
                'new_customers': 'New Customers', 'spend': 'Spend', 'impression': 'Impressions',
                'clicks': 'Clicks', 'ctr': 'CTR', 'cpc': 'CPC', 'roas': 'ROAS'
            })

            
            correlation_matrix = correlation_df.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale=px.colors.diverging.Tropic, 
                zmid=0, 
                text=correlation_matrix.values,
                texttemplate="%{text:.2f}",
                hoverongaps=False
            ))

            st.plotly_chart(fig_corr, use_container_width=True)

            with st.expander("How to Read This Heatmap ❔"):
                st.markdown("""
                    This heatmap visualizes the statistical relationship between different metrics. The values range from -1 to +1:

                    * **Strong Positive (close to +1.0, dark green):** When one metric goes up, the other tends to go up as well. For example, a high correlation between **Spend** and **Revenue** is desirable.
                    * **Strong Negative (close to -1.0, dark purple):** When one metric goes up, the other tends to go down. For example, you might expect to see a negative correlation between **CPC (Cost Per Click)** and **Profit**.
                    * **Weak (close to 0.0, light color):** There is little to no linear relationship between the two metrics.
                    """)

            fig_corr.update_layout(
                title="Correlation Matrix of Marketing & Business KPIs",
                height=600,
                template='plotly_dark', 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                yaxis_autorange='reversed' 
            )

df = load_and_prepare_data()

# --- MAIN APP ---
if df is not None:
    st.title("Marketing Intelligence Dashboard")
    st.markdown("This dashboard provides a complete view of your marketing performance. Select a date range in the sidebar to begin your analysis.")

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Date Range")
    start_date = st.sidebar.date_input(
        "Start Date", 
        df['date'].min(),
        min_value=df['date'].min(),
        max_value=df['date'].max()
    )
    end_date = st.sidebar.date_input(
        "End Date",
        df['date'].max(),
        min_value=df['date'].min(),
        max_value=df['date'].max()
    )
    
    # Platform filter
    platforms = st.sidebar.multiselect(
        "Advertising Platforms",
        options=df['platform'].unique(),
        default=df['platform'].unique()
    )
    
    # Convert dates to datetime64[ns] to match dataframe
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df_filtered = df_filtered[df_filtered['platform'].isin(platforms)]


    # --- TABS FOR ANALYSIS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary", 
        "Channel Analysis", 
        "Strategy & Geographic Analysis", 
        "Business Impact Analysis",
        "Performance Highlights & Intelligent Recommendations",
        "Data Quality & Exports"
    ])

    with tab1:
        st.header("Executive Summary")
        
        # Calculate KPIs
        total_spend = df_filtered['spend'].sum()

        business_summary = df_filtered[['date', 'total_revenue', 'gross_profit', 'new_customers', 'num_orders']].drop_duplicates()

        total_revenue = business_summary['total_revenue'].sum()
        total_profit = business_summary['gross_profit'].sum()
        total_new_customers = business_summary['new_customers'].sum()
        total_orders = business_summary['num_orders'].sum()
        
        # Calculate overall ROAS and CAC safely
        overall_roas = (df_filtered['attributed_revenue'].sum() / total_spend) if total_spend > 0 else 0
        overall_cac = (total_spend / total_new_customers) if total_new_customers > 0 else 0
        overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0

        # Display KPIs in columns
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric(label="Total Spend", value=f"${total_spend:,.0f}")
        kpi2.metric(label="Total Revenue", value=f"${total_revenue:,.0f}")
        kpi3.metric(label="Gross Profit", value=f"${total_profit:,.0f}")
        kpi4.metric(label="Overall ROAS", value=f"{overall_roas:.2f}x")
        kpi5.metric(label="Overall CAC", value=f"${overall_cac:,.2f}")

        # Profitability indicator
        profit_col, margin_col, orders_col = st.columns(3)
        profit_col.metric(label="Profit Margin", value=f"{overall_margin:.1f}%")
        margin_col.metric(label="Total Orders", value=f"{total_orders:,.0f}")
        orders_col.metric(label="New Customers", value=f"{total_new_customers:,.0f}")

        # Performance Trends
        st.subheader("Performance Trends")
        daily_summary = df_filtered.groupby('date').agg({
            'total_revenue': 'mean', 
            'gross_profit': 'mean', 
            'spend': 'sum',
            'num_orders': 'mean',
            'new_customers': 'mean'
        }).reset_index()

        fig_trends = px.line(
            daily_summary,
            x='date',
            y=['total_revenue', 'gross_profit', 'spend'],
            title='Daily Revenue, Profit, and Spend',
            labels={'value': 'Amount (USD)', 'variable': 'Metric', 'date': 'Date'},
            template='plotly_white'
        )
        fig_trends.update_layout(yaxis_title='Amount (USD)', legend_title_text='Metrics')
        st.plotly_chart(fig_trends, use_container_width=True)

    with tab2:
        st.header("Channel Performance Analysis")
        
        # Aggregate data by platform
        platform_summary = df_filtered.groupby('platform').agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'clicks': 'sum',
            'impression': 'sum',
            'new_customers': 'sum',
            'num_orders': 'sum'
        }).reset_index()

        # Calculate metrics for each platform
        platform_summary['roas'] = np.where(
            platform_summary['spend'] > 0,
            platform_summary['attributed_revenue'] / platform_summary['spend'],
            0
        )
        platform_summary['ctr'] = np.where(
            platform_summary['impression'] > 0,
            (platform_summary['clicks'] / platform_summary['impression']) * 100,
            0
        )
        platform_summary['cac'] = np.where(
            platform_summary['new_customers'] > 0,
            platform_summary['spend'] / platform_summary['new_customers'],
            0
        )
        platform_summary['cpc'] = np.where(
            platform_summary['clicks'] > 0,
            platform_summary['spend'] / platform_summary['clicks'],
            0
        )

        st.subheader("Platform Performance Matrix")
        st.markdown("*Interactive bubble chart showing platform efficiency across multiple dimensions*")
        
        fig_matrix = go.Figure()
        
        # Color palette for platforms
        colors = {'Facebook': '#1877F2', 'Google': '#EA4335', 'TikTok': '#8A2BE2'}

        # Icon URLs for each platform
        icon_urls = {
            'Facebook':'https://cdn-icons-png.freepik.com/512/2496/2496095.png',
            'Google': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/1024px-Google_%22G%22_logo.svg.png',
            'TikTok': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ64Be_o28Q7NGSjtAIziMorw31wOr9qauEoQ&s'
        }
        
        x_range = platform_summary['roas'].max() - platform_summary['roas'].min()
        y_range = platform_summary['ctr'].max() - platform_summary['ctr'].min()
        icon_size_factor = 0.175  
        sizex = x_range * icon_size_factor if x_range > 0 else 1
        sizey = y_range * icon_size_factor if y_range > 0 else 1


        for _, platform in platform_summary.iterrows():
            bubble_size = np.sqrt(platform['spend']) / 50 if platform['spend'] > 0 else 5
            bubble_size = max(10, min(bubble_size, 100)) 
            
            fig_matrix.add_trace(go.Scatter(
                x=[platform['roas']],
                y=[platform['ctr']],
                mode='markers', 
                marker=dict(
                    size=0.3,  
                    opacity=0 
                ),
                name=platform['platform'],
                hovertemplate=
                "<b>%{text}</b><br>" +
                "ROAS: %{x:.2f}x<br>" +
                "CTR: %{y:.2f}%<br>" +
                f"Spend: ${platform['spend']:,.0f}<br>" +
                f"CAC: ${platform['cac']:.2f}<br>" +
                f"Orders: {platform['num_orders']:,.0f}<br>" +
                "<extra></extra>",
                text=[platform['platform']],
                showlegend=False
            ))

            fig_matrix.add_layout_image(
                dict(
                    source=icon_urls.get(platform['platform']),
                    xref="x", yref="y",
                    x=platform['roas'], y=platform['ctr'],
                    sizex=sizex, sizey=sizey,
                    xanchor="center", yanchor="middle",
                    sizing="contain",
                    layer="above"
                )
            )
        
        avg_roas = platform_summary['roas'].mean()
        avg_ctr = platform_summary['ctr'].mean()
        

        fig_matrix.add_vline(
            x=avg_roas, 
            line_dash="dash", 
            line_color="gray", 
            opacity=0.5,
            annotation_text=f"Avg ROAS: {avg_roas:.2f}x"
        )
        
    
        fig_matrix.add_hline(
            y=avg_ctr, 
            line_dash="dash", 
            line_color="gray", 
            opacity=0.5,
            annotation_text=f"Avg CTR: {avg_ctr:.2f}%"
        )
        
        max_roas = platform_summary['roas'].max() * 1.1
        max_ctr = platform_summary['ctr'].max() * 1.1
        
        fig_matrix.add_annotation(
            x=max_roas * 0.85, y=max_ctr * 0.85,
            text="🌟 HIGH PERFORMANCE<br>High ROAS + High CTR",
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(144, 238, 144, 0.3)",
            bordercolor="green",
            borderwidth=1
        )
        
        fig_matrix.add_annotation(
            x=avg_roas * 0.5, y=max_ctr * 0.85,
            text="🎯 ENGAGEMENT FOCUS<br>High CTR, Lower ROAS",
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(255, 165, 0, 0.2)",
            bordercolor="orange",
            borderwidth=1
        )
        
        fig_matrix.add_annotation(
            x=max_roas * 0.85, y=avg_ctr * 0.3,
            text="💰 CONVERSION FOCUS<br>High ROAS, Lower CTR",
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(173, 216, 230, 0.3)",
            bordercolor="blue",
            borderwidth=1
        )
        
        fig_matrix.add_annotation(
            x=avg_roas * 0.3, y=avg_ctr * 0.3,
            text="⚠️ NEEDS OPTIMIZATION<br>Below Average",
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(255, 182, 193, 0.3)",
            bordercolor="red",
            borderwidth=1
        )
        
        fig_matrix.update_layout(
            title=dict(
                text="Platform Performance Matrix",
                x=0.5,
                 font=dict(family='Arial', size=16, color='white')
            ),
            xaxis_title="Return on Ad Spend (ROAS)",
            yaxis_title="Click-Through Rate (%)",
            plot_bgcolor='#1E1E1E',  
            paper_bgcolor='#1E1E1E', 
             font=dict(family="Arial", size=12, color='white'),
            height=500,
            margin=dict(l=60, r=60, t=100, b=60),
            hovermode='closest'
        )
        

        fig_matrix.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.2)')
        fig_matrix.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.2)')
        
        st.plotly_chart(fig_matrix, use_container_width=True)
        

        st.subheader("Detailed Platform Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_radar = go.Figure()
            
            max_values = {
                'roas': platform_summary['roas'].max(),
                'ctr': platform_summary['ctr'].max(),
                'cpc_inv': platform_summary['cpc'].min(), 
                'cac_inv': platform_summary['cac'].min() 
            }
            
            for _, platform in platform_summary.iterrows():
                normalized_metrics = [
                    (platform['roas'] / max_values['roas']) * 100 if max_values['roas'] > 0 else 0,
                    (platform['ctr'] / max_values['ctr']) * 100 if max_values['ctr'] > 0 else 0,
                    (max_values['cpc_inv'] / platform['cpc']) * 100 if platform['cpc'] > 0 else 0,
                    (max_values['cac_inv'] / platform['cac']) * 100 if platform['cac'] > 0 else 0
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_metrics + [normalized_metrics[0]], 
                    theta=['ROAS', 'CTR', 'CPC Efficiency', 'CAC Efficiency', 'ROAS'],
                    fill='toself',
                    name=platform['platform'],
                    line_color=colors.get(platform['platform'], '#636EFA'),
                    opacity=0.6
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        ticksuffix="%"
                    )),
                title="Platform Efficiency Radar<br><sub>All metrics normalized to 100%</sub>",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            # Performance trends over time
            daily_platform_perf = df_filtered.groupby(['date', 'platform']).agg({
                'roas': 'mean',
                'spend': 'sum'
            }).reset_index()
            
            fig_trends = px.line(
                daily_platform_perf,
                x='date',
                y='roas',
                color='platform',
                title='Platform ROAS Trends Over Time',
                labels={'roas': 'ROAS (x)', 'date': 'Date'},
                color_discrete_map=colors
            )
            
            fig_trends.update_layout(
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
        
        # Standard comparison charts (enhanced)
        st.subheader("Key Performance Indicators")
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig_roas = px.bar(
                platform_summary,
                x='platform',
                y='roas',
                title='ROAS by Platform',
                labels={'roas': 'Return on Ad Spend (x)', 'platform': 'Platform'},
                color='platform',
                color_discrete_map=colors,
                template='plotly_white',
                text='roas'
            )
            fig_roas.update_traces(texttemplate='%{text:.2f}x', textposition='outside')
            fig_roas.update_layout(showlegend=False)
            st.plotly_chart(fig_roas, use_container_width=True)
            
            fig_ctr = px.bar(
                platform_summary,
                x='platform',
                y='ctr',
                title='CTR (Click-Through Rate) by Platform (%)',
                labels={'ctr': 'Click-Through Rate (%)', 'platform': 'Platform'},
                color='platform',
                color_discrete_map=colors,
                template='plotly_white',
                text='ctr'
            )
            fig_ctr.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_ctr.update_layout(showlegend=False)
            st.plotly_chart(fig_ctr, use_container_width=True)

        with col4:
            fig_spend = px.bar(
                platform_summary,
                x='platform',
                y='spend',
                title='Total Spend by Platform',
                labels={'spend': 'Total Spend (USD)', 'platform': 'Platform'},
                color='platform',
                color_discrete_map=colors,
                template='plotly_white',
                text='spend'
            )
            fig_spend.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_spend.update_layout(showlegend=False)
            st.plotly_chart(fig_spend, use_container_width=True)
            
            fig_cac = px.bar(
                platform_summary,
                x='platform',
                y='cac',
                title='Customer Acquisition Cost by Platform',
                labels={'cac': 'CAC (USD)', 'platform': 'Platform'},
                color='platform',
                color_discrete_map=colors,
                template='plotly_white',
                text='cac'
            )
            fig_cac.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
            fig_cac.update_layout(showlegend=False)
            st.plotly_chart(fig_cac, use_container_width=True)

    with tab3:
        st.header("Campaign Deep Dive")
        
        # Campaign filters
        col1, col2 = st.columns(2)
        with col1:
            campaigns = st.multiselect(
                "Select Campaigns",
                options=df_filtered['campaign'].unique(),
                default=df_filtered['campaign'].unique()[:5] if len(df_filtered['campaign'].unique()) > 5 else df_filtered['campaign'].unique()
            )
        with col2:
            tactics = st.multiselect(
                "Select Tactics",
                options=df_filtered['tactic'].unique(),
                default=df_filtered['tactic'].unique()
            )
        
        campaign_df = df_filtered[
            (df_filtered['campaign'].isin(campaigns)) & 
            (df_filtered['tactic'].isin(tactics))
        ]
        
        campaign_summary = campaign_df.groupby(['platform', 'campaign', 'tactic']).agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'clicks': 'sum',
            'impression': 'sum',
            'new_customers': 'sum'
        }).reset_index()
        
        campaign_summary['roas'] = np.where(
            campaign_summary['spend'] > 0,
            campaign_summary['attributed_revenue'] / campaign_summary['spend'],
            0
        )
        campaign_summary['ctr'] = np.where(
            campaign_summary['impression'] > 0,
            (campaign_summary['clicks'] / campaign_summary['impression']) * 100,
            0
        )
        campaign_summary['cac'] = np.where(
            campaign_summary['new_customers'] > 0,
            campaign_summary['spend'] / campaign_summary['new_customers'],
            0
        )

        st.dataframe(
            campaign_summary.sort_values('roas', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "spend": st.column_config.NumberColumn(format="$%.2f"),
                "attributed_revenue": st.column_config.NumberColumn(format="$%.2f"),
                "roas": st.column_config.NumberColumn(format="%.2fx"),
                "ctr": st.column_config.NumberColumn(format="%.2f%%"),
                "cac": st.column_config.NumberColumn(format="$%.2f"),
            }
        )

        display_geographic_analysis(campaign_df)

        st.markdown("---")
        st.subheader("Tactic Performance Analysis")
        
        if not df_filtered.empty:
            tactic_performance = df_filtered.groupby(['platform', 'tactic']).agg({
                'spend': 'sum',
                'attributed_revenue': 'sum',
                'clicks': 'sum',
                'impression': 'sum',
                'new_customers': 'sum'
            }).reset_index()

            tactic_performance['roas'] = np.where(tactic_performance['spend'] > 0, tactic_performance['attributed_revenue'] / tactic_performance['spend'], 0)
            tactic_performance['ctr'] = np.where(tactic_performance['impression'] > 0, (tactic_performance['clicks'] / tactic_performance['impression']) * 100, 0)
            tactic_performance['cac'] = np.where(tactic_performance['new_customers'] > 0, tactic_performance['spend'] / tactic_performance['new_customers'], 0)

            st.markdown("##### Detailed Tactic Breakdown")
            st.dataframe(
                tactic_performance.sort_values('roas', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "spend": st.column_config.NumberColumn(format="$%.2f"),
                    "attributed_revenue": st.column_config.NumberColumn(format="$%.2f"),
                    "roas": st.column_config.NumberColumn(format="%.2fx"),
                    "ctr": st.column_config.NumberColumn(format="%.2f%%"),
                    "cac": st.column_config.NumberColumn(format="$%.2f"),
                }
            )

            st.markdown("##### Tactic Performance Treemap")
            fig_tactic_treemap = px.treemap(
                tactic_performance,
                path=[px.Constant("All Platforms"), 'platform', 'tactic'],
                values='spend',
                color='roas',
                hover_data=['ctr', 'cac'],
                color_continuous_scale='RdYlGn',
                title='Tactic Performance: Spend (size) vs. ROAS (color)'
            )
            fig_tactic_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            st.plotly_chart(fig_tactic_treemap, use_container_width=True)
        else:
            st.info("No data available for the selected filters to analyze tactics.")

    with tab4:
        st.header("Business Impact Analysis")

        show_customer_growth_metrics(df_filtered)
        st.markdown("---") 
        display_correlation_heatmap(df_filtered)
        
        st.markdown("---")
        st.subheader("Customer Acquisition Efficiency") 
        
        daily_business = df_filtered.groupby('date').agg({
            'spend': 'sum',
            'num_orders': 'sum',
            'new_customers': 'sum',
            'total_revenue': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        
        scatter_col1, scatter_col2 = st.columns(2)
        
        with scatter_col1:
            fig_spend_orders = px.scatter(
                daily_business,
                x='spend',
                y='num_orders',
                title='Marketing Spend vs Orders (with trend)',
                trendline='ols',
                labels={'spend': 'Daily Spend (USD)', 'num_orders': 'Daily Orders'}
            )
            
            if len(daily_business) > 2:
                r_squared = np.corrcoef(daily_business['spend'], daily_business['num_orders'])[0,1]**2
                fig_spend_orders.add_annotation(
                    x=daily_business['spend'].max() * 0.05, y=daily_business['num_orders'].max() * 0.95,
                    text=f"R² = {r_squared:.3f}", showarrow=False,
                    bgcolor="white", bordercolor="gray", borderwidth=1
                )
            
            st.plotly_chart(fig_spend_orders, use_container_width=True)
        
        with scatter_col2:
            fig_customer_growth = px.scatter(
                daily_business,
                x='spend',
                y='new_customers',
                trendline='ols',
                title='Marketing Spend vs New Customers',
                labels={'spend': 'Daily Spend (USD)', 'new_customers': 'Daily New Customers'}
            )
            
            if len(daily_business) > 2:
                r_squared = np.corrcoef(daily_business['spend'], daily_business['new_customers'])[0,1]**2
                fig_customer_growth.add_annotation(
                    x=daily_business['spend'].max() * 0.05, y=daily_business['new_customers'].max() * 0.95,
                    text=f"R² = {r_squared:.3f}", showarrow=False,
                    bgcolor="white", bordercolor="gray", borderwidth=1
                )
            st.plotly_chart(fig_customer_growth, use_container_width=True)


    with tab5:
        st.header("Intelligent Analytics & Recommendations")
        st.markdown("*Automated insights and alerts based on your filtered data's performance patterns.*")

        def generate_intelligent_insights(data):
            """Generate rule-based insights that adapt to current filters"""
            if data.empty:
                return [], [], []

            total_spend = data['spend'].sum()
            total_attributed_revenue = data['attributed_revenue'].sum()
            
            insights = []
            recommendations = []
            alerts = []
            
            platform_perf = data.groupby('platform').agg({
                'roas': 'mean', 'ctr': 'mean', 'cac': 'mean', 'spend': 'sum'
            }).reset_index()
            
            if len(platform_perf) > 1:    
                best_roas = platform_perf.loc[platform_perf['roas'].idxmax()]
                worst_roas = platform_perf.loc[platform_perf['roas'].idxmin()]
                best_ctr = platform_perf.loc[platform_perf['ctr'].idxmax()]
                
                positive_cac = platform_perf[platform_perf['cac'] > 0]
                if not positive_cac.empty:
                    lowest_cac = positive_cac.loc[positive_cac['cac'].idxmin()]
                    if lowest_cac['cac'] < 50:
                        insights.append(f"➠ **Efficient Acquisition:** **{lowest_cac['platform']}** offers efficient customer acquisition at just **${lowest_cac['cac']:.2f} CAC**.")
        
                roas_gap = ((best_roas['roas'] - worst_roas['roas']) / worst_roas['roas'] * 100) if worst_roas['roas'] > 0 else 0
                insights.append(f"➠ **Top Performer:** **{best_roas['platform']}** is outperforming **{worst_roas['platform']}** by **{roas_gap:.0f}%** in ROAS ({best_roas['roas']:.2f}x vs {worst_roas['roas']:.2f}x).")

                if roas_gap > 50:
                    recommendations.append(f"**Budget Allocation:** Consider reallocating 15-20% of the budget from **{worst_roas['platform']}** to **{best_roas['platform']}** to maximize returns.")

                if best_ctr['ctr'] > platform_perf['ctr'].mean() * 1.5:
                    recommendations.append(f"**Creative Strategy:** Analyze **{best_ctr['platform']}'s** high-performing creatives and test them on other platforms.")
                    
            overall_roas = total_attributed_revenue / total_spend if total_spend > 0 else 0
            if overall_roas < 2.0:
                alerts.append(f"⚠️ **Low ROAS Alert:** The current overall ROAS is **{overall_roas:.2f}x**, which may be below your target profitability threshold. A review of underperforming campaigns is recommended.")


            return insights, recommendations, alerts
        
        insights, recommendations, alerts = generate_intelligent_insights(df_filtered)

        if insights or recommendations or alerts:
            if recommendations:
                with st.container(border=True):
                    st.markdown("### 🎯 Strategic Recommendations")
                    for rec in recommendations:
                        st.success(rec)

            if insights:
                with st.container(border=True):
                    st.markdown("### 🔍 Key Insights")
                    for insight in insights:
                        st.info(insight)

            if alerts:
                with st.container(border=True):
                    st.markdown("### ⚠️ Performance Alerts")
                    for alert in alerts:
                        st.warning(alert)

        else:
            st.info("Select a broader date range or more platforms to generate automated insights.")

        st.markdown("---")
        display_top_campaigns(df_filtered)

    with tab6:
        st.header("Data Quality & Monitoring Dashboard")
        st.markdown("*Ensure quality of the data which is being used for getting insights.*")
        
        def calculate_data_quality_metrics(df):
            """Calculate comprehensive data quality metrics"""
            total_records = len(df)
            if total_records == 0:
                return {}
            
            # Data completeness
            completeness_metrics = {}
            key_columns = ['spend', 'clicks', 'impression', 'attributed_revenue', 'total_revenue', 'new_customers']
            
            for col in key_columns:
                if col in df.columns:
                    non_null_count = df[col].notna().sum()
                    completeness_metrics[col] = (non_null_count / total_records) * 100
            
            # Data freshness
            latest_date = df['date'].max()
            earliest_date = df['date'].min()
            days_span = (latest_date - earliest_date).days
            
            # Data consistency checks
            negative_spend = (df['spend'] < 0).sum()
            negative_revenue = (df['total_revenue'] < 0).sum() if 'total_revenue' in df.columns else 0
            impossible_ctr = ((df['clicks'] > df['impression']) & (df['impression'] > 0)).sum()
            
            # Missing attribution data
            zero_attribution = ((df['attributed_revenue'] == 0) & (df['spend'] > 0)).sum()
            
            # Calculate data quality score
            avg_completeness = np.mean(list(completeness_metrics.values())) if completeness_metrics else 0
            consistency_score = max(0, 100 - ((negative_spend + negative_revenue + impossible_ctr) / total_records * 100))
            attribution_score = max(0, 100 - (zero_attribution / total_records * 100)) if total_records > 0 else 100
            
            overall_quality_score = (avg_completeness * 0.4 + consistency_score * 0.3 + attribution_score * 0.3)
            
            return {
                'total_records': total_records,
                'date_range_days': days_span,
                'latest_date': latest_date,
                'earliest_date': earliest_date,
                'completeness_metrics': completeness_metrics,
                'avg_completeness': avg_completeness,
                'negative_spend_count': negative_spend,
                'negative_revenue_count': negative_revenue,
                'impossible_ctr_count': impossible_ctr,
                'zero_attribution_count': zero_attribution,
                'consistency_score': consistency_score,
                'attribution_score': attribution_score,
                'overall_quality_score': overall_quality_score
            }
        
        # Calculate quality metrics for filtered data
        quality_metrics = calculate_data_quality_metrics(df_filtered)
        
        if quality_metrics:
            # Data Quality Summary Cards
            st.markdown("**Data Quality Overview**")
            
            qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
            
            # Quality score with color coding
            quality_score = quality_metrics['overall_quality_score']
            quality_color = "🟢" if quality_score >= 90 else "🟡" if quality_score >= 75 else "🔴"
            
            qual_col1.metric(
                "Overall Data Quality", 
                f"{quality_score:.1f}/100 {quality_color}",
                help="Composite score based on completeness, consistency, and attribution coverage"
            )
            
            qual_col2.metric(
                "Data Completeness", 
                f"{quality_metrics['avg_completeness']:.1f}%",
                help="Average percentage of non-null values across key metrics"
            )
            
            qual_col3.metric(
                "Records Analyzed", 
                f"{quality_metrics['total_records']:,}",
                help="Total number of data points in current filtered view"
            )
            
            # Data freshness indicator
            days_old = (pd.Timestamp.now().date() - quality_metrics['latest_date'].date()).days
            freshness_status = "🟢 Fresh" if days_old <= 1 else "🟡 Recent" if days_old <= 7 else "🔴 Stale"
            
            qual_col4.metric(
                "Data Freshness", 
                f"{days_old} days old",
                delta=freshness_status,
                help=f"Latest data point: {quality_metrics['latest_date'].strftime('%Y-%m-%d')}"
            )
            
            # Detailed Quality Analysis
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                
                if quality_metrics['completeness_metrics']:
                    # Create completeness visualization
                    completeness_df = pd.DataFrame([
                        {'Field': field.replace('_', ' ').title(), 'Completeness': completeness}
                        for field, completeness in quality_metrics['completeness_metrics'].items()
                    ])
                    
                    fig_completeness = px.bar(
                        completeness_df,
                        x='Completeness',
                        y='Field',
                        orientation='h',
                        title='Data Completeness by Field (%)',
                        color='Completeness',
                        color_continuous_scale=['red', 'yellow', 'green'],
                        range_color=[0, 100]
                    )
                    
                    st.plotly_chart(fig_completeness, use_container_width=True)
            
            with detail_col2:
                st.markdown("**⚠️ Data Quality Issues**")
                
                # Issue tracking
                issues = []
                
                if quality_metrics['negative_spend_count'] > 0:
                    issues.append(f"🔴 {quality_metrics['negative_spend_count']} records with negative spend")
                
                if quality_metrics['negative_revenue_count'] > 0:
                    issues.append(f"🔴 {quality_metrics['negative_revenue_count']} records with negative revenue")
                
                if quality_metrics['impossible_ctr_count'] > 0:
                    issues.append(f"🟡 {quality_metrics['impossible_ctr_count']} records with clicks > impressions")
                
                if quality_metrics['zero_attribution_count'] > 0:
                    zero_attribution_pct = (quality_metrics['zero_attribution_count'] / quality_metrics['total_records']) * 100
                    if zero_attribution_pct > 10:
                        issues.append(f"🟡 {zero_attribution_pct:.1f}% of spend has zero attributed revenue")
                
                if quality_metrics['avg_completeness'] < 95:
                    issues.append(f"🟡 Data completeness below 95% ({quality_metrics['avg_completeness']:.1f}%)")
                
                if days_old > 7:
                    issues.append(f"🔴 Data is {days_old} days old - consider refreshing")
                
                if not issues:
                    st.success("✅ No significant data quality issues detected!")
                else:
                    for issue in issues:
                        st.warning(issue)

            display_export_options(df_filtered)

            st.markdown("---")

            st.metric(
                "Data Sources",
                f"{len(df['platform'].unique())} platforms",
                help="Facebook, Google, TikTok + Business data"
            )
        


else:
    st.error("Data could not be loaded. Please ensure the 'data' folder and required CSV files exist.")