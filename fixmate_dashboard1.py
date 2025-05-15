# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="FixMate Home Services Dashboard",
    page_icon="🔧",
    layout="wide"
)

# Function to generate sample data if no file is provided
def generate_sample_data():
    """
    Generate sample data for demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample dataframe with FixMate Home Services data
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Define service categories, technicians, and client types
    service_categories = ['Plumbing', 'Electrical', 'HVAC', 'Appliance Repair', 'Roofing']
    technicians = ['Smith, J.', 'Johnson, M.', 'Williams, R.', 'Brown, T.', 'Garcia, L.', 'Martinez, D.']
    client_types = ['New', 'Returning', 'Referred']
    
    # Generate date range (last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate hours for each day
    hours = [f"{h:02d}:00:00" for h in range(6, 22)]  # 6 AM to 9 PM
    
    # Create lists to store data
    records = []
    
    # Generate sample data with specific patterns
    for date in dates:
        for time in hours:
            for service in service_categories:
                # Skip some combinations to reduce dataset size
                if np.random.rand() > 0.15:
                    continue
                
                # Ad spend varies by service and time of day
                base_ad_spend = np.random.randint(50, 200)
                hour = int(time.split(':')[0])
                
                # Higher spend during work hours
                if 8 <= hour <= 18:
                    ad_spend = base_ad_spend * 1.5
                else:
                    ad_spend = base_ad_spend
                
                # Conversions vary based on service popularity and time
                base_conversions = np.random.randint(0, 8)
                
                # Peak hours have higher conversion rates
                time_factor = 1.8 if 9 <= hour <= 11 or 17 <= hour <= 19 else 0.8
                
                # Plumbing and HVAC are more popular
                service_factor = 1.5 if service in ['Plumbing', 'HVAC'] else 1.0
                
                # Weekend factor
                weekday = date.weekday()
                weekend_factor = 1.2 if weekday >= 5 else 1.0  # Higher on weekends
                
                conversions = max(0, int(base_conversions * time_factor * service_factor * weekend_factor))
                
                # Revenue varies by service type and conversion count
                if conversions > 0:
                    base_revenue_per_conversion = {
                        'Plumbing': np.random.randint(150, 350),
                        'Electrical': np.random.randint(120, 250),
                        'HVAC': np.random.randint(200, 450),
                        'Appliance Repair': np.random.randint(80, 200),
                        'Roofing': np.random.randint(350, 800)
                    }
                    revenue = conversions * base_revenue_per_conversion[service]
                else:
                    revenue = 0
                
                # Client type distribution
                if conversions > 0:
                    # Returning clients more common for maintenance services
                    if service in ['HVAC', 'Appliance Repair']:
                        client_type_weights = {'New': 0.3, 'Returning': 0.5, 'Referred': 0.2}
                    else:
                        client_type_weights = {'New': 0.5, 'Returning': 0.3, 'Referred': 0.2}
                    
                    client_type = np.random.choice(list(client_type_weights.keys()), 
                                                  p=list(client_type_weights.values()))
                else:
                    client_type = None
                
                # Add random variance for realism
                ad_spend = max(10, ad_spend + np.random.randint(-20, 20))
                revenue = max(0, revenue + np.random.randint(-50, 50) if revenue > 0 else 0)
                
                # Assign a technician if there were conversions
                technician = np.random.choice(technicians) if conversions > 0 else None
                
                # Create record
                record = {
                    'Date': date.strftime('%Y-%m-%d'),
                    'Time': time,
                    'Service Category': service,
                    'Technician': technician,
                    'Ad Spend': round(ad_spend, 2),
                    'Conversions': conversions,
                    'Revenue': revenue,
                    'Client Type': client_type
                }
                records.append(record)
    
    # Create dataframe
    df = pd.DataFrame(records)
    
    # Filter out rows where there's no conversion data to make dataset smaller
    df = df[~((df['Conversions'] == 0) & (np.random.rand(len(df)) < 0.7))]
    
    return df

# Load and prepare the dataset
def load_and_prepare_data(uploaded_file=None):
    """
    Load the FixMate Home Services dataset and prepare it for analysis.
    
    Parameters:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pd.DataFrame: Prepared dataframe
    """
    # Load data or generate sample data
    if uploaded_file is not None:
        try:
            df = pd.read_csv('FixMate_Home_Services.csv')
            st.success("✅ File successfully loaded!")
        except Exception as e:
            st.error(f"Error loading file: {e}. Using sample data instead.")
            df = generate_sample_data()
    else:
        df = generate_sample_data()
        st.info("ℹ️ Using sample data. Upload your own CSV file for custom analysis.")
    
    # Convert Date and Time columns to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Show time samples
    st.write("Sample Time values:")
    st.write(df['Time'].head())
    
    # Convert safely
    df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
    
    # Handle parsing errors
    if df['Hour'].isna().all():
        st.error("❌ Could not parse the 'Time' column. Please check the time format in your CSV.")
        st.stop()
    
    
    # Create datetime column combining Date and Time
    df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'])
    
    # Create ROI column (avoid division by zero)
    df['ROI'] = df.apply(lambda row: row['Revenue'] / row['Ad Spend'] if row['Ad Spend'] > 0 else 0, axis=1)
    
    # Create customer value column (avoid division by zero)
    df['Customer Value'] = df.apply(lambda row: row['Revenue'] / row['Conversions'] if row['Conversions'] > 0 else 0, axis=1)
    
    # Create day of week
    df['DayOfWeek'] = df['Date'].dt.day_name()
    
    # Create month column
    df['Month'] = df['Date'].dt.strftime('%Y-%m')
    
    return df

# Analyze high-value services
def analyze_high_value_services(df):
    """
    Determine which services attract the highest value customers over time.
    
    Parameters:
        df (pd.DataFrame): Prepared dataframe
        
    Returns:
        tuple: (avg_value_by_service_df, value_trend_df, client_type_pct)
    """
    # Calculate average customer value by service category
    avg_value_by_service = df.groupby('Service Category')['Customer Value'].mean().reset_index()
    avg_value_by_service = avg_value_by_service.sort_values('Customer Value', ascending=False)
    
    # Calculate customer value trend over time by service category
    value_trend = df.groupby(['Month', 'Service Category'])['Customer Value'].mean().reset_index()
    value_trend['Month'] = pd.to_datetime(value_trend['Month'] + '-01')
    
    # Calculate client type distribution by service
    client_type_dist = df.groupby(['Service Category', 'Client Type']).size().unstack(fill_value=0)
    client_type_pct = client_type_dist.div(client_type_dist.sum(axis=1), axis=0) * 100
    client_type_pct_df = client_type_pct.reset_index().melt(
        id_vars='Service Category',
        var_name='Client Type',
        value_name='Percentage'
    )
    
    return avg_value_by_service, value_trend, client_type_pct_df

# Analyze optimal ad delivery windows
def analyze_optimal_ad_windows(df):
    """
    Analyze customer acquisition by hour to find optimal ad delivery windows.
    
    Parameters:
        df (pd.DataFrame): Prepared dataframe
        
    Returns:
        tuple: Multiple dataframes with hourly analysis
    """
    # Analyze conversions by hour
    hourly_conversions = df.groupby('Hour')['Conversions'].sum().reset_index()
    
    # Analyze ROI by hour
    hourly_roi = df.groupby('Hour')['ROI'].mean().reset_index()
    
    # Analyze conversions by hour and service category
    hourly_service = df.groupby(['Hour', 'Service Category'])['Conversions'].sum().reset_index()
    
    # Analyze conversions by hour and day of week
    hourly_day = df.groupby(['Hour', 'DayOfWeek'])['Conversions'].sum().reset_index()
    
    # Calculate ad efficiency (conversions per ad dollar) by hour
    df['Conv_per_Dollar'] = df['Conversions'] / df['Ad Spend'].replace(0, np.nan)
    hourly_efficiency = df.groupby('Hour')['Conv_per_Dollar'].mean().reset_index()
    
    return hourly_conversions, hourly_roi, hourly_service, hourly_day, hourly_efficiency

# Analyze ad spend impact
def analyze_ad_spend_impact(df):
    """
    Analyze the impact of ad spend on conversions and revenue.
    
    Parameters:
        df (pd.DataFrame): Prepared dataframe
        
    Returns:
        tuple: Correlation values and aggregated dataframes
    """
    # Calculate correlation between ad spend and conversions/revenue
    corr_spend_conv = df['Ad Spend'].corr(df['Conversions'])
    corr_spend_rev = df['Ad Spend'].corr(df['Revenue'])
    
    # Aggregate data by service category
    service_metrics = df.groupby('Service Category').agg({
        'Ad Spend': 'sum',
        'Conversions': 'sum',
        'Revenue': 'sum',
        'ROI': 'mean',
        'Customer Value': 'mean'
    }).reset_index()
    
    # Calculate conversion rate
    service_metrics['Conversion Rate'] = service_metrics['Conversions'] / service_metrics['Ad Spend']
    
    return corr_spend_conv, corr_spend_rev, service_metrics

# Generate insights and recommendations
def generate_insights(df, avg_value_by_service, hourly_conversions, hourly_roi):
    """
    Generate insights and recommendations based on the analysis.
    
    Returns:
        dict: Dictionary with insights and recommendations
    """
    # Top services by customer value
    top_services = avg_value_by_service.head(3)['Service Category'].tolist()
    top_service_values = avg_value_by_service.head(3)['Customer Value'].tolist()
    
    # Top hours by conversions
    top_hours_conv = hourly_conversions.sort_values('Conversions', ascending=False).head(3)['Hour'].tolist()
    
    # Top hours by ROI
    top_hours_roi = hourly_roi.sort_values('ROI', ascending=False).head(3)['Hour'].tolist()
    
    # Calculate service growth
    df['Month'] = df['Date'].dt.to_period('M')
    service_growth = df.groupby(['Month', 'Service Category'])['Conversions'].sum().unstack()
    
    try:
        service_growth_pct = service_growth.pct_change().iloc[-1].sort_values(ascending=False)
        fastest_growing = service_growth_pct.index[0]
        growth_rate = service_growth_pct.iloc[0]
    except:
        fastest_growing = "Insufficient data"
        growth_rate = 0
    
    # Client type analysis
    client_type_value = df.groupby('Client Type')['Customer Value'].mean().sort_values(ascending=False)
    
    insights = {
        'top_services': [f"{s} (${v:.2f})" for s, v in zip(top_services, top_service_values)],
        'fastest_growing': fastest_growing,
        'growth_rate': growth_rate,
        'top_hours_conv': [f"{h}:00" for h in top_hours_conv],
        'top_hours_roi': [f"{h}:00" for h in top_hours_roi],
        'client_insights': {ct: f"${val:.2f}" for ct, val in client_type_value.items()}
    }
    
    return insights

# Main Streamlit dashboard
def main():
    st.title("🔧 FixMate Home Services Acquisition Dashboard")
    st.markdown("""
    This dashboard helps evaluate how service type, campaign timing, and ad spend influence customer sign-ups 
    for urgent home repairs. Upload your CSV file or use sample data to analyze:
    - **Which services attract the highest value customers over time**
    - **Optimal hours for ad delivery to maximize conversions**
    """)
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your FixMate Home Services CSV", type=['csv'])
    
    # Load and prepare data
    df = load_and_prepare_data(uploaded_file)
    
    # Show sample of the data
    with st.expander("📊 Preview Data", expanded=False):
        st.write(df.head())
        
        st.markdown("### Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        col3.metric("Total Revenue", f"${df['Revenue'].sum():,.2f}")
        
        # Display key metrics
        total_conv = df['Conversions'].sum()
        total_spend = df['Ad Spend'].sum()
        overall_roi = df['Revenue'].sum() / total_spend if total_spend > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Conversions", f"{total_conv:,}")
        col2.metric("Total Ad Spend", f"${total_spend:,.2f}")
        col3.metric("Overall ROI", f"{overall_roi:.2f}x")
    
    # Run analyses
    avg_value_by_service, value_trend, client_type_pct = analyze_high_value_services(df)
    hourly_conversions, hourly_roi, hourly_service, hourly_day, hourly_efficiency = analyze_optimal_ad_windows(df)
    corr_spend_conv, corr_spend_rev, service_metrics = analyze_ad_spend_impact(df)
    insights = generate_insights(df, avg_value_by_service, hourly_conversions, hourly_roi)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 High-Value Services", 
        "🕒 Optimal Ad Windows", 
        "💰 Ad Spend Impact",
        "🔍 Insights & Recommendations"
    ])
    
    # Tab 1: High-Value Services
    with tab1:
        st.header("Services by Customer Value")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for average customer value by service
            fig_avg_value = px.bar(
                avg_value_by_service,
                x='Service Category',
                y='Customer Value',
                title='Average Customer Value by Service Category',
                color='Customer Value',
                color_continuous_scale='Viridis',
                text_auto='.2f'
            )
            fig_avg_value.update_layout(
                xaxis_title='Service Category', 
                yaxis_title='Average Customer Value ($)',
                height=400
            )
            st.plotly_chart(fig_avg_value, use_container_width=True)
        
        with col2:
            # Stacked bar chart for client type distribution by service
            fig_client_type = px.bar(
                client_type_pct,
                x='Service Category',
                y='Percentage',
                color='Client Type',
                title='Client Type Distribution by Service Category (%)',
                barmode='stack',
                text_auto='.1f'
            )
            fig_client_type.update_layout(
                xaxis_title='Service Category', 
                yaxis_title='Percentage (%)',
                height=400
            )
            st.plotly_chart(fig_client_type, use_container_width=True)
        
        # Line chart for customer value trend over time
        fig_value_trend = px.line(
            value_trend,
            x='Month',
            y='Customer Value',
            color='Service Category',
            title='Customer Value Trend by Service Category Over Time',
            markers=True
        )
        fig_value_trend.update_layout(
            xaxis_title='Month', 
            yaxis_title='Average Customer Value ($)',
            height=400
        )
        st.plotly_chart(fig_value_trend, use_container_width=True)
        
        # Display service metrics table
        st.subheader("Service Performance Metrics")
        formatted_metrics = service_metrics.copy()
        formatted_metrics['Ad Spend'] = formatted_metrics['Ad Spend'].map('${:,.2f}'.format)
        formatted_metrics['Revenue'] = formatted_metrics['Revenue'].map('${:,.2f}'.format)
        formatted_metrics['ROI'] = formatted_metrics['ROI'].map('{:.2f}x'.format)
        formatted_metrics['Customer Value'] = formatted_metrics['Customer Value'].map('${:,.2f}'.format)
        formatted_metrics['Conversion Rate'] = formatted_metrics['Conversion Rate'].map('{:.4f}'.format)
        st.dataframe(formatted_metrics, use_container_width=True)
    
    # Tab 2: Optimal Ad Windows
    with tab2:
        st.header("Optimal Ad Delivery Windows")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for conversions by hour
            fig_hourly_conv = px.bar(
                hourly_conversions,
                x='Hour',
                y='Conversions',
                title='Total Conversions by Hour of Day',
                color='Conversions',
                color_continuous_scale='Viridis',
                text_auto=True
            )
            fig_hourly_conv.update_layout(
                xaxis_title='Hour of Day (24h)', 
                yaxis_title='Total Conversions',
                height=400
            )
            st.plotly_chart(fig_hourly_conv, use_container_width=True)
        
        with col2:
            # Bar chart for ROI by hour
            fig_hourly_roi = px.bar(
                hourly_roi,
                x='Hour',
                y='ROI',
                title='Average ROI by Hour of Day',
                color='ROI',
                color_continuous_scale='RdYlGn',
                text_auto='.2f'
            )
            fig_hourly_roi.update_layout(
                xaxis_title='Hour of Day (24h)', 
                yaxis_title='Average ROI',
                height=400
            )
            st.plotly_chart(fig_hourly_roi, use_container_width=True)
        
        # Create heatmap for conversions by hour and day of week
        pivot_hourly_day = hourly_day.pivot(index='DayOfWeek', columns='Hour', values='Conversions')
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        try:
            pivot_hourly_day = pivot_hourly_day.reindex(day_order)
        except:
            pass  # Handle case when not all days are in the data
            
        fig_hourly_day = px.imshow(
            pivot_hourly_day,
            title='Conversions by Hour and Day of Week',
            color_continuous_scale='Viridis',
            labels=dict(x="Hour of Day (24h)", y="Day of Week", color="Conversions"),
            text_auto=True
        )
        fig_hourly_day.update_layout(height=450)
        st.plotly_chart(fig_hourly_day, use_container_width=True)
        
        # Create heatmap for conversions by hour and service category
        pivot_hourly_service = hourly_service.pivot(index='Service Category', columns='Hour', values='Conversions')
        fig_hourly_service = px.imshow(
            pivot_hourly_service,
            title='Conversions by Hour and Service Category',
            color_continuous_scale='Viridis',
            labels=dict(x="Hour of Day (24h)", y="Service Category", color="Conversions"),
            text_auto=True
        )
        fig_hourly_service.update_layout(height=400)
        st.plotly_chart(fig_hourly_service, use_container_width=True)
        
        # Line chart for ad efficiency by hour
        fig_efficiency = px.line(
            hourly_efficiency,
            x='Hour',
            y='Conv_per_Dollar',
            title='Ad Efficiency (Conversions per Ad Dollar Spent) by Hour',
            markers=True
        )
        fig_efficiency.update_layout(
            xaxis_title='Hour of Day (24h)', 
            yaxis_title='Conversions per Dollar Spent',
            height=400
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Tab 3: Ad Spend Impact
    with tab3:
        st.header("Ad Spend Impact Analysis")
        
        # Metrics row
        col1, col2 = st.columns(2)
        col1.metric("Ad Spend to Conversions Correlation", f"{corr_spend_conv:.3f}")
        col2.metric("Ad Spend to Revenue Correlation", f"{corr_spend_rev:.3f}")
        
        # Create scatter plot of ad spend vs conversions
        fig_spend_conv = px.scatter(
            df,
            x='Ad Spend',
            y='Conversions',
            color='Service Category',
            size='Revenue',
            hover_data=['Date', 'ROI'],
            title='Ad Spend vs. Conversions by Service Category',
            trendline='ols'
        )
        fig_spend_conv.update_layout(
            xaxis_title='Ad Spend ($)', 
            yaxis_title='Conversions',
            height=450
        )
        st.plotly_chart(fig_spend_conv, use_container_width=True)
        
        # Create scatter plot of ad spend vs revenue
        fig_spend_rev = px.scatter(
            df,
            x='Ad Spend',
            y='Revenue',
            color='Service Category',
            size='Conversions',
            hover_data=['Date', 'ROI'],
            title='Ad Spend vs. Revenue by Service Category',
            trendline='ols'
        )
        fig_spend_rev.update_layout(
            xaxis_title='Ad Spend ($)', 
            yaxis_title='Revenue ($)',
            height=450
        )
        st.plotly_chart(fig_spend_rev, use_container_width=True)
        
        # Service ROI comparison
        service_roi = df.groupby('Service Category')['ROI'].mean().reset_index().sort_values('ROI', ascending=False)
        fig_service_roi = px.bar(
            service_roi,
            x='Service Category',
            y='ROI',
            title='Average ROI by Service Category',
            color='ROI',
            color_continuous_scale='RdYlGn',
            text_auto='.2f'
        )
        fig_service_roi.update_layout(
            xaxis_title='Service Category', 
            yaxis_title='Average ROI',
            height=400
        )
        st.plotly_chart(fig_service_roi, use_container_width=True)
    
    # Tab 4: Insights & Recommendations
    with tab4:
        st.header("Key Insights & Recommendations")
        
        # Display insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("High-Value Services")
            st.markdown(f"""
            **Top services by customer value:**
            1. {insights['top_services'][0] if len(insights['top_services']) > 0 else 'N/A'}
            2. {insights['top_services'][1] if len(insights['top_services']) > 1 else 'N/A'}
            3. {insights['top_services'][2] if len(insights['top_services']) > 2 else 'N/A'}
            
            **Fastest growing service:** {insights['fastest_growing']} ({insights['growth_rate']:.1%} growth)
            """)
            
            st.subheader("Client Type Analysis")
            for client_type, value in insights['client_insights'].items():
                st.markdown(f"- **{client_type} clients:** Average value {value}")
        
        with col2:
            st.subheader("Optimal Ad Windows")
            st.markdown(f"""
            **Top hours for conversions:**
            1. {insights['top_hours_conv'][0] if len(insights['top_hours_conv']) > 0 else 'N/A'}
            2. {insights['top_hours_conv'][1] if len(insights['top_hours_conv']) > 1 else 'N/A'}
            3. {insights['top_hours_conv'][2] if len(insights['top_hours_conv']) > 2 else 'N/A'}
            
            **Top hours for ROI:**
            1. {insights['top_hours_roi'][0] if len(insights['top_hours_roi']) > 0 else 'N/A'}
            2. {insights['top_hours_roi'][1] if len(insights['top_hours_roi']) > 1 else 'N/A'}
            3. {insights['top_hours_roi'][2] if len(insights['top_hours_roi']) > 2 else 'N/A'}
            """)
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("Strategic Recommendations")
        
        rec1, rec2, rec3 = st.columns(3)
        
        with rec1:
            st.markdown("#### Service Focus")
            st.info("""
            **Prioritize marketing the highest value services:**
            - Allocate higher budget percentages to top-performing services
            - Create service-specific campaigns for the top 3 services identified
            - Test premium pricing strategies for high-value services
            """)
        
        with rec2:
            st.markdown("#### Timing Optimization")
            st.info("""
            **Adjust ad delivery timing:**
            - Schedule campaigns during peak conversion hours
            - Set higher bids during hours with best ROI
            - Create dayparting strategies based on the heatmap patterns
            - Test weekend vs. weekday strategies for different services
            """)
        
        with rec3:
            st.markdown("#### Client Type Targeting")
            st.info("""
            **Customize approaches by client type:**
            - Develop retention programs for high-value returning clients
            - Create referral incentives for highest-value segments
            - Design acquisition funnels optimized for new customer types
            - Implement client-specific follow-up processes
            """)
        
        # Action items
        st.subheader("Immediate Action Items")
        st.success("""
        1. **Reallocate ad budget**: Shift 20% of budget from lowest to highest ROI services
        2. **Optimize ad schedules**: Configure campaigns to focus delivery during top 3 ROI hours
        3. **Analyze technician performance**: Evaluate conversion rates by technician to identify training needs
        4. **Service bundling**: Test service combinations based on client overlap analysis
        5. **Data enrichment**: Add geographic and demographic data to enable location-based optimization
        """)

# Run the Streamlit app
if __name__ == "__main__":
    main()
