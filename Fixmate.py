import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="FixMate Home Services - Acquisition Dashboard",
    page_icon="🔧",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #34495e;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔧 FixMate Home Services - Acquisition Dashboard</div>', unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        # Load the actual CSV file
        df = pd.read_csv('FixMate_Home_Services.csv')
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract month and day of week for additional analysis
        df['Month'] = df['Date'].dt.month_name()
        df['Day_of_Week'] = df['Date'].dt.day_name()
        
        # Calculate conversion rate and cost per conversion
        df['Conversion_Rate'] = df['Conversions'] / 1  # Assuming each row is a campaign instance
        df['Cost_Per_Conversion'] = df['Ad Spend'] / df['Conversions'].replace(0, np.nan)
        df['Revenue_Per_Conversion'] = df['Revenue'] / df['Conversions'].replace(0, np.nan)
        df['ROI'] = (df['Revenue'] - df['Ad Spend']) / df['Ad Spend'] * 100
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create a sample dataframe with the structure shown in the example
        columns = ['Date', 'Time', 'Service Category', 'Technician', 'Ad Spend', 
                  'Conversions', 'Revenue', 'Client Type']
        return pd.DataFrame(columns=columns)

# Load the data
df = load_data()

# Sidebar for filters
st.sidebar.markdown('## Filters')

# Date range filter
min_date = df['Date'].min() if not df.empty else pd.Timestamp('2024-01-01')
max_date = df['Date'].max() if not df.empty else pd.Timestamp('2025-05-01')
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

# Time of day filter
time_options = df['Time'].unique().tolist() if not df.empty else []
selected_times = st.sidebar.multiselect(
    'Time of Day',
    options=time_options,
    default=time_options
)

if selected_times:
    filtered_df = filtered_df[filtered_df['Time'].isin(selected_times)]

# Service category filter
service_options = df['Service Category'].unique().tolist() if not df.empty else []
selected_services = st.sidebar.multiselect(
    'Service Categories',
    options=service_options,
    default=service_options
)

if selected_services:
    filtered_df = filtered_df[filtered_df['Service Category'].isin(selected_services)]

# Client type filter
client_options = df['Client Type'].unique().tolist() if not df.empty else []
selected_clients = st.sidebar.multiselect(
    'Client Types',
    options=client_options,
    default=client_options
)

if selected_clients:
    filtered_df = filtered_df[filtered_df['Client Type'].isin(selected_clients)]

# Main dashboard content
st.markdown('## Dashboard Overview')

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Conversions", 
        value=f"{filtered_df['Conversions'].sum():,.0f}",
        delta=None
    )

with col2:
    total_spend = filtered_df['Ad Spend'].sum()
    st.metric(
        label="Total Ad Spend", 
        value=f"${total_spend:,.2f}",
        delta=None
    )

with col3:
    total_revenue = filtered_df['Revenue'].sum()
    st.metric(
        label="Total Revenue",
        value=f"${total_revenue:,.2f}",
        delta=None
    )

with col4:
    avg_cost_per_conversion = total_spend / filtered_df['Conversions'].sum() if filtered_df['Conversions'].sum() > 0 else 0
    st.metric(
        label="Avg. Cost per Conversion",
        value=f"${avg_cost_per_conversion:,.2f}",
        delta=None
    )

# Create tabs for different analysis views
tab1, tab2, tab3, tab4 = st.tabs(["Service Analysis", "Timing Analysis", "Cost Analysis", "Client Analysis"])

with tab1:
    st.markdown('### Service Category Performance')
    
    # Group by Service Category
    service_perf = filtered_df.groupby('Service Category').agg({
        'Conversions': 'sum',
        'Ad Spend': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    service_perf['Cost_Per_Conversion'] = service_perf['Ad Spend'] / service_perf['Conversions']
    service_perf['ROI'] = (service_perf['Revenue'] - service_perf['Ad Spend']) / service_perf['Ad Spend'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            service_perf,
            x='Service Category',
            y='Conversions',
            color='Service Category',
            title='Conversions by Service Category'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            service_perf,
            x='Service Category',
            y='Cost_Per_Conversion',
            color='Service Category',
            title='Cost per Conversion by Service Category'
        )
        fig2.update_layout(yaxis_title='Cost per Conversion ($)')
        st.plotly_chart(fig2, use_container_width=True)

    # ROI by service category
    fig3 = px.bar(
        service_perf,
        x='Service Category',
        y='ROI',
        color='Service Category',
        title='Return on Investment (ROI) by Service Category'
    )
    fig3.update_layout(yaxis_title='ROI (%)')
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown('### Timing Analysis')
    
    # Time of day analysis
    time_perf = filtered_df.groupby('Time').agg({
        'Conversions': 'sum',
        'Ad Spend': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    time_perf['Conversion_Rate'] = time_perf['Conversions'] / time_perf['Conversions'].sum()
    time_perf['Cost_Per_Conversion'] = time_perf['Ad Spend'] / time_perf['Conversions']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig4 = px.bar(
            time_perf,
            x='Time',
            y='Conversions',
            color='Time',
            title='Conversions by Time of Day'
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        fig5 = px.bar(
            time_perf,
            x='Time',
            y='Cost_Per_Conversion',
            color='Time',
            title='Cost per Conversion by Time of Day'
        )
        fig5.update_layout(yaxis_title='Cost per Conversion ($)')
        st.plotly_chart(fig5, use_container_width=True)
    
    # Service category performance by time of day
    time_service_perf = filtered_df.groupby(['Time', 'Service Category']).agg({
        'Conversions': 'sum'
    }).reset_index()
    
    fig6 = px.bar(
        time_service_perf,
        x='Time',
        y='Conversions',
        color='Service Category',
        title='Service Category Performance by Time of Day',
        barmode='group'
    )
    st.plotly_chart(fig6, use_container_width=True)

with tab3:
    st.markdown('### Ad Spend Analysis')
    
    # Create Ad Spend buckets for analysis
    filtered_df['Spend_Bucket'] = pd.cut(
        filtered_df['Ad Spend'],
        bins=[0, 50, 100, 150, float('inf')],
        labels=['$0-50', '$50-100', '$100-150', '$150+']
    )
    
    spend_perf = filtered_df.groupby('Spend_Bucket').agg({
        'Conversions': 'sum',
        'Ad Spend': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    spend_perf['ROI'] = (spend_perf['Revenue'] - spend_perf['Ad Spend']) / spend_perf['Ad Spend'] * 100
    spend_perf['Conversion_Rate'] = spend_perf['Conversions'] / spend_perf['Conversions'].sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig7 = px.bar(
            spend_perf,
            x='Spend_Bucket',
            y='Conversions',
            color='Spend_Bucket',
            title='Conversions by Ad Spend Range'
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        fig8 = px.bar(
            spend_perf,
            x='Spend_Bucket',
            y='ROI',
            color='Spend_Bucket',
            title='ROI by Ad Spend Range'
        )
        fig8.update_layout(yaxis_title='ROI (%)')
        st.plotly_chart(fig8, use_container_width=True)
    
    # Scatter plot of Ad Spend vs Conversions
    fig9 = px.scatter(
        filtered_df,
        x='Ad Spend',
        y='Conversions',
        color='Service Category',
        size='Revenue',
        hover_data=['Time', 'Client Type', 'Revenue'],
        title='Ad Spend vs. Conversions by Service Category'
    )
    st.plotly_chart(fig9, use_container_width=True)

with tab4:
    st.markdown('### Client Analysis')
    
    # Client type performance
    client_perf = filtered_df.groupby('Client Type').agg({
        'Conversions': 'sum',
        'Ad Spend': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    client_perf['Revenue_Per_Client'] = client_perf['Revenue'] / client_perf['Conversions']
    client_perf['ROI'] = (client_perf['Revenue'] - client_perf['Ad Spend']) / client_perf['Ad Spend'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig10 = px.pie(
            client_perf,
            values='Conversions',
            names='Client Type',
            title='Conversion Distribution by Client Type'
        )
        st.plotly_chart(fig10, use_container_width=True)

    with col2:
        fig11 = px.bar(
            client_perf,
            x='Client Type',
            y='Revenue_Per_Client',
            color='Client Type',
            title='Revenue per Client by Client Type'
        )
        fig11.update_layout(yaxis_title='Revenue per Client ($)')
        st.plotly_chart(fig11, use_container_width=True)
    
    # Service Category preference by Client Type
    service_client = filtered_df.groupby(['Client Type', 'Service Category']).agg({
        'Conversions': 'sum'
    }).reset_index()
    
    fig12 = px.bar(
        service_client,
        x='Client Type',
        y='Conversions',
        color='Service Category',
        title='Service Category Preference by Client Type',
        barmode='group'
    )
    st.plotly_chart(fig12, use_container_width=True)

# Key Insights Section
st.markdown('## Key Insights')

# Only show insights if we have data
if not filtered_df.empty:
    # Get top performing service
    top_service = filtered_df.groupby('Service Category')['Conversions'].sum().idxmax()
    top_service_conversions = filtered_df.groupby('Service Category')['Conversions'].sum().max()
    
    # Get best time of day
    best_time = filtered_df.groupby('Time')['Conversions'].sum().idxmax()
    best_time_conversions = filtered_df.groupby('Time')['Conversions'].sum().max()
    
    # Get most valuable client type
    most_valuable_client = filtered_df.groupby('Client Type')['Revenue'].sum().idxmax()
    most_valuable_client_revenue = filtered_df.groupby('Client Type')['Revenue'].sum().max()
    
    # Get optimal ad spend range
    if 'Spend_Bucket' in filtered_df.columns:
        optimal_spend = filtered_df.groupby('Spend_Bucket')['ROI'].mean().idxmax()
    else:
        optimal_spend = "Not enough data"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">' +
                   f'<h3>Top Performing Service: {top_service}</h3>' +
                   f'<p>Generated {top_service_conversions:.0f} conversions during the selected period.</p>' +
                   '</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="insight-box">' +
                   f'<h3>Optimal Ad Delivery Time: {best_time}</h3>' +
                   f'<p>Generated {best_time_conversions:.0f} conversions during the selected period.</p>' +
                   '</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">' +
                   f'<h3>Most Valuable Client Type: {most_valuable_client}</h3>' +
                   f'<p>Generated ${most_valuable_client_revenue:.2f} in revenue during the selected period.</p>' +
                   '</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="insight-box">' +
                   f'<h3>Optimal Ad Spend Range: {optimal_spend}</h3>' +
                   f'<p>This spend range produced the highest return on investment.</p>' +
                   '</div>', unsafe_allow_html=True)

# Show the raw data if requested
with st.expander("View Raw Data"):
    st.dataframe(filtered_df)

# Add a download button for the filtered data
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="fixmate_filtered_data.csv",
    mime="text/csv",
)
