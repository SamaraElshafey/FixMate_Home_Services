import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="FixMate High-Value Customer Analysis",
    page_icon="💎",
    layout="wide"
)

# Page title
st.title("💎 FixMate Home Services - High-Value Customer Analysis")
st.markdown("This dashboard analyzes which services attract the highest value customers over time.")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('FixMate_Home_Services.csv')
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract month and quarter for time-based analysis
        df['Month'] = df['Date'].dt.month_name()
        df['Month_Num'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Quarter_Label'] = 'Q' + df['Quarter'].astype(str) + ' ' + df['Date'].dt.year.astype(str)
        
        # Calculate key metrics for value analysis
        df['Revenue_Per_Conversion'] = df['Revenue'] / df['Conversions']
        df['Profit'] = df['Revenue'] - df['Ad Spend']
        df['ROI'] = df['Profit'] / df['Ad Spend'] * 100
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['Date'] >= pd.Timestamp(start_date)) & 
                         (df['Date'] <= pd.Timestamp(end_date))]
    else:
        filtered_df = df.copy()

    # Client type filter
    client_types = ['All'] + sorted(df['Client Type'].unique().tolist())
    selected_client_type = st.sidebar.selectbox("Client Type", client_types)
    
    if selected_client_type != 'All':
        filtered_df = filtered_df[filtered_df['Client Type'] == selected_client_type]
    
    # Define high-value customers
    value_metric = st.sidebar.radio(
        "Define High-Value by:",
        ["Revenue Per Conversion", "Total Revenue", "Client Type (Loyal/Returning)"]
    )
    
    # Main Analysis
    st.header("Service Analysis for High-Value Customers")
    
    # Create high-value customer segment based on selected metric
    if value_metric == "Revenue Per Conversion":
        # Define high-value threshold (top 25% of revenue per conversion)
        high_value_threshold = filtered_df['Revenue_Per_Conversion'].quantile(0.75)
        filtered_df['High_Value'] = filtered_df['Revenue_Per_Conversion'] >= high_value_threshold
        st.info(f"High-value customers defined as those with revenue per conversion ≥ ${high_value_threshold:.2f}")
    
    elif value_metric == "Total Revenue":
        # Group by service and get revenue
        high_value_threshold = filtered_df['Revenue'].quantile(0.75)
        filtered_df['High_Value'] = filtered_df['Revenue'] >= high_value_threshold
        st.info(f"High-value customers defined as those with total revenue ≥ ${high_value_threshold:.2f}")
    
    else:  # Client Type
        filtered_df['High_Value'] = filtered_df['Client Type'].isin(['Loyal', 'Returning'])
        st.info("High-value customers defined as those with 'Loyal' or 'Returning' status")
    
    # Create columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_value_count = filtered_df[filtered_df['High_Value']].shape[0]
        total_count = filtered_df.shape[0]
        high_value_pct = (high_value_count / total_count) * 100 if total_count > 0 else 0
        
        st.metric(
            "High-Value Customers",
            f"{high_value_count} ({high_value_pct:.1f}%)"
        )
    
    with col2:
        high_value_revenue = filtered_df[filtered_df['High_Value']]['Revenue'].sum()
        total_revenue = filtered_df['Revenue'].sum()
        high_value_rev_pct = (high_value_revenue / total_revenue) * 100 if total_revenue > 0 else 0
        
        st.metric(
            "Revenue from High-Value",
            f"${high_value_revenue:.2f} ({high_value_rev_pct:.1f}%)"
        )
    
    with col3:
        avg_rev_high = filtered_df[filtered_df['High_Value']]['Revenue_Per_Conversion'].mean()
        avg_rev_all = filtered_df['Revenue_Per_Conversion'].mean()
        
        st.metric(
            "Avg. Revenue Per Conversion (High-Value)",
            f"${avg_rev_high:.2f}",
            delta=f"{(avg_rev_high - avg_rev_all):.2f}"
        )
    
    # Analysis 1: Which services attract high-value customers
    st.subheader("Services Attracting High-Value Customers")
    
    # Group by Service Category to see distribution of high-value customers
    service_high_value = filtered_df.groupby('Service Category')['High_Value'].sum().reset_index()
    service_high_value.rename(columns={'High_Value': 'High_Value_Count'}, inplace=True)
    
    # Get total counts per service
    service_counts = filtered_df.groupby('Service Category').size().reset_index(name='Total_Count')
    
    # Merge the two dataframes
    service_analysis = pd.merge(service_high_value, service_counts, on='Service Category')
    service_analysis['High_Value_Percentage'] = (service_analysis['High_Value_Count'] / service_analysis['Total_Count']) * 100
    
    # Calculate additional metrics for each service
    service_revenue = filtered_df.groupby('Service Category')['Revenue'].sum().reset_index()
    service_analysis = pd.merge(service_analysis, service_revenue, on='Service Category')
    
    # Sort by high-value percentage
    service_analysis = service_analysis.sort_values('High_Value_Percentage', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart showing high-value customer percentage by service
        fig = px.bar(
            service_analysis,
            x='Service Category',
            y='High_Value_Percentage',
            color='High_Value_Percentage',
            color_continuous_scale='Viridis',
            title='Percentage of High-Value Customers by Service'
        )
        fig.update_layout(yaxis_title='High-Value Customers (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart showing total revenue by service
        fig = px.bar(
            service_analysis,
            x='Service Category',
            y='Revenue',
            color='High_Value_Percentage',
            color_continuous_scale='Viridis',
            title='Total Revenue by Service'
        )
        fig.update_layout(yaxis_title='Revenue ($)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis 2: High-value customers over time
    st.subheader("High-Value Customer Trends Over Time")
    
    # Group by month/quarter
    time_period = st.radio("Time Period", ["Monthly", "Quarterly"])
    
    if time_period == "Monthly":
        time_col = 'Month'
        sort_col = 'Month_Num'
        # Get monthly data
        time_analysis = filtered_df.groupby(['Month', 'Month_Num', 'Service Category'])['High_Value'].sum().reset_index()
        time_analysis.rename(columns={'High_Value': 'High_Value_Count'}, inplace=True)
        
        # Get total counts
        time_counts = filtered_df.groupby(['Month', 'Month_Num', 'Service Category']).size().reset_index(name='Total_Count')
        
    else:  # Quarterly
        time_col = 'Quarter_Label'
        sort_col = 'Quarter'
        # Get quarterly data
        time_analysis = filtered_df.groupby(['Quarter_Label', 'Quarter', 'Service Category'])['High_Value'].sum().reset_index()
        time_analysis.rename(columns={'High_Value': 'High_Value_Count'}, inplace=True)
        
        # Get total counts
        time_counts = filtered_df.groupby(['Quarter_Label', 'Quarter', 'Service Category']).size().reset_index(name='Total_Count')
    
    # Merge the two dataframes
    time_analysis = pd.merge(time_analysis, time_counts, on=[time_col, sort_col, 'Service Category'])
    time_analysis['High_Value_Percentage'] = (time_analysis['High_Value_Count'] / time_analysis['Total_Count']) * 100
    
    # Sort by time period
    time_analysis = time_analysis.sort_values(sort_col)
    
    # Line chart showing high-value customer trends over time by service
    fig = px.line(
        time_analysis,
        x=time_col,
        y='High_Value_Percentage',
        color='Service Category',
        markers=True,
        title=f'High-Value Customer Percentage by Service Over Time ({time_period})'
    )
    fig.update_layout(yaxis_title='High-Value Customers (%)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis 3: Client type distribution by service
    st.subheader("Client Type Distribution for High-Value Services")
    
    # Client type distribution for high-value customers by service
    client_service = filtered_df[filtered_df['High_Value']].groupby(['Service Category', 'Client Type']).size().reset_index(name='Count')
    
    # Calculate percentages
    client_service_totals = client_service.groupby('Service Category')['Count'].sum().reset_index()
    client_service_totals.rename(columns={'Count': 'Total'}, inplace=True)
    client_service = pd.merge(client_service, client_service_totals, on='Service Category')
    client_service['Percentage'] = (client_service['Count'] / client_service['Total']) * 100
    
    # Bar chart showing client type distribution for high-value customers by service
    fig = px.bar(
        client_service,
        x='Service Category',
        y='Count',
        color='Client Type',
        title='Client Type Distribution for High-Value Customers by Service',
        text='Percentage',
        barmode='stack'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis 4: Service performance metrics for high-value customers
    st.subheader("Service Performance Metrics for High-Value Customers")
    
    # Calculate key metrics by service for high-value customers
    hv_metrics = filtered_df[filtered_df['High_Value']].groupby('Service Category').agg({
        'Conversions': 'sum',
        'Revenue': 'sum',
        'Ad Spend': 'sum',
        'Revenue_Per_Conversion': 'mean',
        'Profit': 'sum',
        'ROI': 'mean'
    }).reset_index()
    
    # Format for display
    hv_metrics['Revenue_Per_Conversion'] = hv_metrics['Revenue_Per_Conversion'].round(2)
    hv_metrics['ROI'] = hv_metrics['ROI'].round(2)
    
    # Display metrics table
    st.dataframe(hv_metrics, use_container_width=True)
    
    # Visualization for ROI and Revenue per Conversion
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            hv_metrics,
            x='Service Category',
            y='ROI',
            color='Service Category',
            title='ROI by Service for High-Value Customers'
        )
        fig.update_layout(yaxis_title='ROI (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            hv_metrics,
            x='Service Category',
            y='Revenue_Per_Conversion',
            color='Service Category',
            title='Revenue Per Conversion by Service for High-Value Customers'
        )
        fig.update_layout(yaxis_title='Revenue Per Conversion ($)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis 5: Time of day analysis for high-value services
    st.subheader("Optimal Time of Day for High-Value Acquisitions")
    
    # Group by time of day and service
    time_hv = filtered_df[filtered_df['High_Value']].groupby(['Time', 'Service Category']).size().reset_index(name='High_Value_Count')
    
    # Get total counts
    time_total = filtered_df.groupby(['Time', 'Service Category']).size().reset_index(name='Total_Count')
    
    # Merge the two dataframes
    time_hv_analysis = pd.merge(time_hv, time_total, on=['Time', 'Service Category'])
    time_hv_analysis['High_Value_Percentage'] = (time_hv_analysis['High_Value_Count'] / time_hv_analysis['Total_Count']) * 100
    
    # Bar chart showing high-value customer percentage by time of day and service
    fig = px.bar(
        time_hv_analysis,
        x='Time',
        y='High_Value_Percentage',
        color='Service Category',
        title='High-Value Customer Percentage by Time of Day and Service',
        barmode='group'
    )
    fig.update_layout(yaxis_title='High-Value Customers (%)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights Section
    st.header("Key Insights")
    
    # Get top service by high-value percentage
    top_service_hv_pct = service_analysis.iloc[0]['Service Category']
    top_service_hv_pct_value = service_analysis.iloc[0]['High_Value_Percentage']
    
    # Get top service by revenue from high-value customers
    top_service_revenue = hv_metrics.sort_values('Revenue', ascending=False).iloc[0]['Service Category']
    top_service_revenue_value = hv_metrics.sort_values('Revenue', ascending=False).iloc[0]['Revenue']
    
    # Get best time of day for high-value acquisitions
    best_time_hv = time_hv_analysis.sort_values('High_Value_Percentage', ascending=False).iloc[0]['Time']
    best_time_hv_service = time_hv_analysis.sort_values('High_Value_Percentage', ascending=False).iloc[0]['Service Category']
    best_time_hv_pct = time_hv_analysis.sort_values('High_Value_Percentage', ascending=False).iloc[0]['High_Value_Percentage']
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Top Services for High-Value Customers
        
        - **Highest Percentage**: {top_service_hv_pct} ({top_service_hv_pct_value:.1f}%)
        - **Highest Revenue**: {top_service_revenue} (${top_service_revenue_value:.2f})
        """)
    
    with col2:
        st.markdown(f"""
        ### Optimal Time for High-Value Acquisitions
        
        - **Best Time**: {best_time_hv} for {best_time_hv_service} ({best_time_hv_pct:.1f}%)
        """)
    
    # Client type insights
    most_loyal_service = client_service[client_service['Client Type'] == 'Loyal'].sort_values('Percentage', ascending=False).iloc[0]['Service Category']
    most_returning_service = client_service[client_service['Client Type'] == 'Returning'].sort_values('Percentage', ascending=False).iloc[0]['Service Category']
    most_new_service = client_service[client_service['Client Type'] == 'New'].sort_values('Percentage', ascending=False).iloc[0]['Service Category']
    
    st.markdown(f"""
    ### Client Type Insights
    
    - **Most Loyal Customers**: {most_loyal_service}
    - **Most Returning Customers**: {most_returning_service}
    - **Most New Customers**: {most_new_service}
    """)
    
    # Show the data table
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)

else:
    st.error("Failed to load data. Please check that your CSV file is properly formatted and accessible.")