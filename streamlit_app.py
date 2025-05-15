# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import calendar

# STEP 1: SET UP THE STREAMLIT APP STRUCTURE
# ==========================================
def main():
    # Configure the page layout and title
    st.set_page_config(
        page_title="FixMate Acquisition Dashboard",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add a title and description
    st.title("🔧 FixMate Home Services - Acquisition Dashboard")
    st.markdown("""
    This dashboard analyzes how service type, campaign timing, and ad spend influence customer sign-ups
    for urgent home repairs. Use the sidebar to navigate through different sections of the application.
    """)
    
    # Create sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "📊 Campaign Performance", "🔍 Service Analysis", "💰 Ad Spend ROI", "📈 Time Analysis"]
    )
    
    # Load data for all pages
    data = load_data()
    
    # Display the selected page
    if page == "🏠 Home":
        home_page(data)
    elif page == "📊 Campaign Performance":
        campaign_performance_page(data)
    elif page == "🔍 Service Analysis":
        service_analysis_page(data)
    elif page == "💰 Ad Spend ROI":
        ad_spend_roi_page(data)
    elif page == "📈 Time Analysis":
        time_analysis_page(data)

# STEP 2: DATA LOADING FUNCTION
# ============================
@st.cache_data
def load_data():
    """Load and cache the FixMate services dataset"""
    try:
        # Try to load data from a CSV file if it exists
        data = pd.read_csv('FixMate_Home_Services.csv')
        st.success("✅ Loaded real data successfully!")
    except FileNotFoundError:
        # If file doesn't exist, create sample data
        st.warning("Sample data is being used. Replace with your actual dataset.")
        np.random.seed(42)
        n_samples = 1000
        
        # Generate dates ranging over a year
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        dates = np.random.choice(dates, n_samples)
        
        # Generate times throughout the day
        hours = np.random.randint(0, 24, n_samples)
        minutes = np.random.choice([0, 15, 30, 45], n_samples)
        times = [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]
        
        # Service categories
        service_categories = ['Plumbing', 'Electrical', 'HVAC', 'Appliance Repair', 'Roofing']
        service_probs = [0.3, 0.25, 0.2, 0.15, 0.1]  # More common services have higher probability
        
        # Technicians
        technicians = ['John Smith', 'Maria Garcia', 'David Johnson', 'Sarah Lee', 
                      'Robert Chen', 'Emily Davis', 'Michael Brown', 'Lisa Wilson']
        
        # Client types
        client_types = ['New', 'Returning']
        
        # Create the sample dataframe
        data = pd.DataFrame({
            'Date': dates,
            'Time': times,
            'Service Category': np.random.choice(service_categories, n_samples, p=service_probs),
            'Technician': np.random.choice(technicians, n_samples),
            'Ad Spend': np.random.uniform(50, 500, n_samples),  # Ad spend between $50-$500
            'Conversions': np.random.randint(0, 10, n_samples),  # Number of conversions
            'Revenue': np.random.uniform(100, 2000, n_samples),  # Revenue between $100-$2000
            'Client Type': np.random.choice(client_types, n_samples, p=[0.7, 0.3])  # 70% new, 30% returning
        })
        
        # Make correlations more realistic
        # Higher ad spend generally leads to more conversions
        high_spend_idx = data['Ad Spend'] > 300
        data.loc[high_spend_idx, 'Conversions'] = np.random.randint(3, 10, sum(high_spend_idx))
        
        # More conversions generally lead to more revenue
        data['Revenue'] = data['Conversions'] * np.random.uniform(150, 300, n_samples)
        
        # Format the date column properly
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        data['Date'] = data['Date'].astype(str)
    
    # Additional preprocessing
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.day_name()
    data['Hour'] = pd.to_datetime(data['Time']).dt.hour
    
    # Calculate ROI
    data['ROI'] = (data['Revenue'] - data['Ad Spend']) / data['Ad Spend']
    
    # Calculate Cost per Conversion
    data.loc[data['Conversions'] > 0, 'Cost per Conversion'] = data['Ad Spend'] / data['Conversions']
    data.loc[data['Conversions'] == 0, 'Cost per Conversion'] = data['Ad Spend']  # Avoid division by zero
    
    return data

# STEP 3: CREATE THE HOME PAGE
# ============================
def home_page(data):
    st.header("🏠 Welcome to the FixMate Acquisition Dashboard")
    
    st.write("""
    ### About This Dashboard
    
    This dashboard helps marketing and operations teams analyze how various factors affect customer acquisition 
    for FixMate Home Services. By exploring the relationships between service types, campaign timing, 
    and ad spend, you can optimize your marketing strategies to maximize customer sign-ups and revenue.
    
    ### How to Use This Dashboard
    
    1. Navigate to **Campaign Performance** to see overall metrics and KPIs
    2. Visit **Service Analysis** to understand which service categories perform best
    3. Check **Ad Spend ROI** to analyze return on investment across campaigns
    4. Explore **Time Analysis** to identify optimal timing for ad campaigns
    """)
    
    # Key metrics
    st.subheader("📈 Key Metrics Overview")
    
    # Create four columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_spend = data['Ad Spend'].sum()
    total_conversions = data['Conversions'].sum()
    total_revenue = data['Revenue'].sum()
    avg_roi = ((data['Revenue'].sum() - data['Ad Spend'].sum()) / data['Ad Spend'].sum()) * 100
    
    with col1:
        st.metric("Total Ad Spend", f"${total_spend:,.2f}")
    
    with col2:
        st.metric("Total Conversions", f"{total_conversions:,}")
    
    with col3:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col4:
        st.metric("Overall ROI", f"{avg_roi:.2f}%")
    
    # Top performing services
    st.subheader("🏆 Top Performing Service Categories")
    
    # Group by service category and calculate key metrics
    service_performance = data.groupby('Service Category').agg({
        'Ad Spend': 'sum',
        'Conversions': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    # Calculate ROI
    service_performance['ROI'] = ((service_performance['Revenue'] - service_performance['Ad Spend']) 
                                / service_performance['Ad Spend']) * 100
    
    # Calculate Cost per Conversion
    service_performance['Cost per Conversion'] = service_performance['Ad Spend'] / service_performance['Conversions']
    
    # Sort by conversions
    service_performance = service_performance.sort_values('Conversions', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        service_performance, 
        x='Service Category', 
        y='Conversions',
        color='ROI',
        color_continuous_scale='RdYlGn',
        text_auto=True,
        title='Service Categories by Conversions and ROI'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show recent performance trend
    st.subheader("📅 Recent Performance Trend")
    
    # Group data by date and calculate daily metrics
    daily_data = data.groupby(data['Date'].dt.date).agg({
        'Ad Spend': 'sum',
        'Conversions': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    # Create a two-line chart for Ad Spend and Conversions
    fig = go.Figure()
    
    # Add Ad Spend line
    fig.add_trace(go.Scatter(
        x=daily_data['Date'], 
        y=daily_data['Ad Spend'],
        name='Ad Spend ($)',
        line=dict(color='#3498db', width=2)
    ))
    
    # Add Conversions line with separate y-axis
    fig.add_trace(go.Scatter(
        x=daily_data['Date'], 
        y=daily_data['Conversions'],
        name='Conversions',
        line=dict(color='#e74c3c', width=2),
        yaxis='y2'
    ))
    
    # Update layout with second y-axis
    fig.update_layout(
        title='Daily Ad Spend vs. Conversions',
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='Ad Spend ($)',
            titlefont=dict(color='#3498db'),
            tickfont=dict(color='#3498db')
        ),
        yaxis2=dict(
            title='Conversions',
            titlefont=dict(color='#e74c3c'),
            tickfont=dict(color='#e74c3c'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        height=400,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# STEP 4: CREATE THE CAMPAIGN PERFORMANCE PAGE
# ===========================================
def campaign_performance_page(data):
    st.header("📊 Campaign Performance Analysis")
    
    # Filters
    st.subheader("🔍 Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range filter
        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()
        
        date_range = st.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = data[(data['Date'].dt.date >= start_date) & (data['Date'].dt.date <= end_date)]
        else:
            filtered_data = data
    
    with col2:
        # Service category filter
        all_categories = data['Service Category'].unique().tolist()
        selected_categories = st.multiselect(
            "Select Service Categories",
            options=all_categories,
            default=all_categories
        )
        
        if selected_categories:
            filtered_data = filtered_data[filtered_data['Service Category'].isin(selected_categories)]
    
    with col3:
        # Client type filter
        all_client_types = data['Client Type'].unique().tolist()
        selected_client_types = st.multiselect(
            "Select Client Types",
            options=all_client_types,
            default=all_client_types
        )
        
        if selected_client_types:
            filtered_data = filtered_data[filtered_data['Client Type'].isin(selected_client_types)]
    
    # Overall metrics
    st.subheader("📈 Overall Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_spend = filtered_data['Ad Spend'].sum()
    total_conversions = filtered_data['Conversions'].sum()
    total_revenue = filtered_data['Revenue'].sum()
    
    with col1:
        st.metric(
            "Total Ad Spend", 
            f"${total_spend:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Total Conversions", 
            f"{total_conversions:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Total Revenue", 
            f"${total_revenue:,.2f}",
            delta=None
        )
    
    with col4:
        if total_spend > 0:
            roi = ((total_revenue - total_spend) / total_spend) * 100
            st.metric(
                "Overall ROI", 
                f"{roi:.2f}%",
                delta=None
            )
        else:
            st.metric("Overall ROI", "N/A", delta=None)
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if total_conversions > 0:
            cpc = total_spend / total_conversions
            st.metric(
                "Cost per Conversion", 
                f"${cpc:.2f}",
                delta=None
            )
        else:
            st.metric("Cost per Conversion", "N/A", delta=None)
    
    with col2:
        if total_conversions > 0:
            avg_revenue = total_revenue / total_conversions
            st.metric(
                "Avg. Revenue per Conversion", 
                f"${avg_revenue:.2f}",
                delta=None
            )
        else:
            st.metric("Avg. Revenue per Conversion", "N/A", delta=None)
    
    with col3:
        new_clients = filtered_data[filtered_data['Client Type'] == 'New']['Conversions'].sum()
        new_pct = (new_clients / total_conversions * 100) if total_conversions > 0 else 0
        st.metric(
            "New Client %", 
            f"{new_pct:.2f}%",
            delta=None
        )
    
    with col4:
        returning_clients = filtered_data[filtered_data['Client Type'] == 'Returning']['Conversions'].sum()
        returning_pct = (returning_clients / total_conversions * 100) if total_conversions > 0 else 0
        st.metric(
            "Returning Client %", 
            f"{returning_pct:.2f}%",
            delta=None
        )
    
    # Campaign performance over time
    st.subheader("📅 Campaign Performance Over Time")
    
    # Group by date
    daily_metrics = filtered_data.groupby(filtered_data['Date'].dt.date).agg({
        'Ad Spend': 'sum',
        'Conversions': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    # Calculate daily ROI
    daily_metrics['ROI'] = ((daily_metrics['Revenue'] - daily_metrics['Ad Spend']) / 
                          daily_metrics['Ad Spend']) * 100
    
    # Create tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["Ad Spend", "Conversions", "Revenue", "ROI"])
    
    with tab1:
        fig = px.line(
            daily_metrics, 
            x='Date', 
            y='Ad Spend',
            title='Daily Ad Spend',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(
            daily_metrics, 
            x='Date', 
            y='Conversions',
            title='Daily Conversions',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.line(
            daily_metrics, 
            x='Date', 
            y='Revenue',
            title='Daily Revenue',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = px.line(
            daily_metrics, 
            x='Date', 
            y='ROI',
            title='Daily ROI (%)',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance by client type
    st.subheader("👥 Performance by Client Type")
    
    # Group by client type
    client_metrics = filtered_data.groupby('Client Type').agg({
        'Ad Spend': 'sum',
        'Conversions': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    # Calculate ROI and CPC
    client_metrics['ROI'] = ((client_metrics['Revenue'] - client_metrics['Ad Spend']) / 
                           client_metrics['Ad Spend']) * 100
    client_metrics['CPC'] = client_metrics['Ad Spend'] / client_metrics['Conversions']
    
    # Add visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            client_metrics,
            x='Client Type',
            y='Conversions',
            title='Conversions by Client Type',
            color='Client Type',
            text_auto=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            client_metrics,
            x='Client Type',
            y='ROI',
            title='ROI by Client Type (%)',
            color='Client Type',
            text_auto=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance data table
    st.subheader("📋 Detailed Performance Data")
    
    st.dataframe(
        client_metrics.style.format({
            'Ad Spend': '${:,.2f}',
            'Revenue': '${:,.2f}',
            'ROI': '{:.2f}%',
            'CPC': '${:.2f}'
        }),
        hide_index=True
    )

# STEP 5: CREATE THE SERVICE ANALYSIS PAGE
# =======================================
def service_analysis_page(data):
    st.header("🔍 Service Category Analysis")
    
    # Filters
    st.subheader("🔍 Filter Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date range filter
        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()
        
        date_range = st.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = data[(data['Date'].dt.date >= start_date) & (data['Date'].dt.date <= end_date)]
        else:
            filtered_data = data
    
    with col2:
        # Client type filter
        all_client_types = data['Client Type'].unique().tolist()
        selected_client_types = st.multiselect(
            "Select Client Types",
            options=all_client_types,
            default=all_client_types
        )
        
        if selected_client_types:
            filtered_data = filtered_data[filtered_data['Client Type'].isin(selected_client_types)]
    
    # Performance by service category
    st.subheader("📊 Service Category Performance")
    
    # Group by service category
    service_metrics = filtered_data.groupby('Service Category').agg({
        'Ad Spend': 'sum',
        'Conversions': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    # Calculate ROI and CPC
    service_metrics['ROI'] = ((service_metrics['Revenue'] - service_metrics['Ad Spend']) / 
                            service_metrics['Ad Spend']) * 100
    service_metrics['CPC'] = service_metrics['Ad Spend'] / service_metrics['Conversions']
    
    # Create visualization
    fig = px.bar(
        service_metrics.sort_values('Conversions', ascending=False),
        x='Service Category',
        y='Conversions',
        color='ROI',
        color_continuous_scale='RdYlGn',
        text_auto=True,
        title='Service Categories by Conversions with ROI Color Scale'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Service category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Ad spend distribution
        fig = px.pie(
            service_metrics,
            values='Ad Spend',
            names='Service Category',
            title='Ad Spend Distribution by Service Category',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue distribution
        fig = px.pie(
            service_metrics,
            values='Revenue',
            names='Service Category',
            title='Revenue Distribution by Service Category',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.subheader("📈 Detailed Service Category Metrics")
    
    # Sort by conversions
    service_metrics_sorted = service_metrics.sort_values('Conversions', ascending=False)
    
    st.dataframe(
        service_metrics_sorted.style.format({
            'Ad Spend': '${:,.2f}',
            'Revenue': '${:,.2f}',
            'ROI': '{:.2f}%',
            'CPC': '${:.2f}'
        }),
        hide_index=True
    )
    
    # Service category trends over time
    st.subheader("📅 Service Category Trends Over Time")
    
    # Group by date and service category
    service_time_data = filtered_data.groupby([filtered_data['Date'].dt.to_period('M'), 'Service Category']).agg({
        'Conversions': 'sum'
    }).reset_index()
    
    # Convert period to string
    service_time_data['Month'] = service_time_data['Date'].dt.to_timestamp().dt.strftime('%Y-%m')
    
    # Create line chart
    fig = px.line(
        service_time_data,
        x='Month',
        y='Conversions',
        color='Service Category',
        title='Monthly Conversions by Service Category',
        markers=True
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Service performance by client type
    st.subheader("👥 Service Performance by Client Type")
    
    # Group by service category and client type
    service_client_data = filtered_data.groupby(['Service Category', 'Client Type']).agg({
        'Conversions': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    # Create grouped bar chart
    fig = px.bar(
        service_client_data,
        x='Service Category',
        y='Conversions',
        color='Client Type',
        barmode='group',
        title='Conversions by Service Category and Client Type',
        text_auto=True
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# STEP 6: CREATE THE AD SPEND ROI PAGE
# ==================================
def ad_spend_roi_page(data):
    st.header("💰 Ad Spend ROI Analysis")
    
    # Filters
    st.subheader("🔍 Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range filter
        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()
        
        date_range = st.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = data[(data['Date'].dt.date >= start_date) & (data['Date'].dt.date <= end_date)]
        else:
            filtered_data = data
    
    with col2:
        # Service category filter
        all_categories = data['Service Category'].unique().tolist()
        selected_categories = st.multiselect(
            "Select Service Categories",
            options=all_categories,
            default=all_categories
        )
        
        if selected_categories:
            filtered_data = filtered_data[filtered_data['Service Category'].isin(selected_categories)]
    
    with col3:
        # Client type filter
        all_client_types = data['Client Type'].unique().tolist()
        selected_client_types = st.multiselect(
            "Select Client Types",
            options=all_client_types,
            default=all_client_types
        )
        
        if selected_client_types:
            filtered_data = filtered_data[filtered_data['Client Type'].isin(selected_client_types)]
    
    # Key ROI metrics
    st.subheader("📊 Key ROI Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_spend = filtered_data['Ad Spend'].sum()
    total_revenue = filtered_data['Revenue'].sum()
    total_conversions = filtered_data['Conversions'].sum()
    
    with col1:
        roi = ((total_revenue - total_spend) / total_spend) * 100 if total_spend > 0 else 0
        st.metric(
            "Overall ROI", 
            f"{roi:.2f}%",
            delta=None
        )
    
    with col2:
        profit = total_revenue - total_spend
        st.metric(
            "Total Profit", 
            f"${profit:,.2f}",
            delta=None
        )
    
    with col3:
        cpc = total_spend / total_conversions if total_conversions > 0 else 0
        st.metric(
            "Cost per Conversion", 
            f"${cpc:.2f}",
            delta=None
        )
    
    with col4:
        roas = total_revenue / total_spend if total_spend > 0 else 0
        st.metric(
            "ROAS", 
            f"{roas:.2f}x",
            delta=None
        )
    
    # Ad spend vs ROI scatter plot
    st.subheader("📈 Ad Spend vs. ROI Relationship")
    
    # Group by date for scatter plot
    daily_data = filtered_data.groupby(filtered_data['Date'].dt.date).agg({
        'Ad Spend': 'sum',
        'Revenue': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    
    # Calculate ROI
    daily_data['ROI'] = ((daily_data['Revenue'] - daily_data['Ad Spend']) / 
                        daily_data['Ad Spend']) * 100
    
    # Create scatter plot
    fig = px.scatter(
        daily_data,
        x='Ad Spend',
        y='ROI',
        size='Conversions',
        color='Conversions',
        hover_name='Date',
        title='Daily Ad Spend vs. ROI',
        labels={'Ad Spend': 'Daily Ad Spend ($)', 'ROI': 'ROI (%)'},
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI by service category
    st.subheader("📊 ROI by Service Category")
    
    # Group by service category
    service_roi = filtered_data.groupby('Service Category').agg({
        'Ad Spend': 'sum',
        'Revenue': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    
    # Calculate metrics
    service_roi['ROI'] = ((service_roi['Revenue'] - service_roi['Ad Spend']) / 
                         service_roi['Ad Spend']) * 100
    service_roi['CPC'] = service_roi['Ad Spend'] / service_roi['Conversions']
    service_roi['ROAS'] = service_roi['Revenue'] / service_roi['Ad Spend']
    
    # Sort by ROI for chart
    service_roi_sorted = service_roi.sort_values('ROI', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        service_roi_sorted,
        x='Service Category',
        y='ROI',
        color='ROI',
        color_continuous_scale='RdYlGn',
        text_auto=True,
        title='ROI by Service Category (%)'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    