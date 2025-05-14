import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 FixMate Acquisition Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("fixmate_home_services.csv")

    # Convert date and time
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df['Datetime'] = df.apply(lambda row: datetime.combine(row['Date'], row['Time']), axis=1)

    # Add time features
    df['Hour'] = df['Datetime'].dt.hour
    df['Weekday'] = df['Datetime'].dt.day_name()
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
selected_services = st.sidebar.multiselect("Select Service Categories", options=df['Service Category'].unique(), default=df['Service Category'].unique())
selected_client_types = st.sidebar.multiselect("Select Client Types", options=df['Client Type'].unique(), default=df['Client Type'].unique())

filtered_df = df[df['Service Category'].isin(selected_services) & df['Client Type'].isin(selected_client_types)]

# Milestone 1: High-value services
st.subheader("🛠️ High-Value Services Over Time")

service_stats = filtered_df.groupby('Service Category').agg({
    'Conversions': 'sum',
    'Revenue': 'sum',
    'Ad Spend': 'sum'
}).reset_index()

service_stats['Revenue per Conversion'] = service_stats['Revenue'] / service_stats['Conversions']
service_stats['ROI'] = service_stats['Revenue'] / service_stats['Ad Spend']

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(data=service_stats.sort_values(by='Revenue per Conversion', ascending=False),
            y='Service Category', x='Revenue per Conversion', palette='viridis', ax=ax1)
ax1.set_title("Revenue per Conversion by Service Category")
st.pyplot(fig1)

# Milestone 2: Hourly performance
st.subheader("⏰ Optimal Ad Delivery Windows")

hourly_stats = filtered_df.groupby('Hour').agg({
    'Conversions': 'sum',
    'Ad Spend': 'sum',
    'Revenue': 'sum'
}).reset_index()

hourly_stats['Revenue per Conversion'] = hourly_stats['Revenue'] / hourly_stats['Conversions']

fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=hourly_stats, x='Hour', y='Conversions', marker='o', ax=ax2)
ax2.set_title("Conversions by Hour")
ax2.set_xticks(range(0, 24))
ax2.grid(True)
st.pyplot(fig2)

# Optional Heatmap
st.subheader("🗓️ Heatmap of Conversions by Hour & Day")

heatmap_data = filtered_df.groupby(['Weekday', 'Hour'])['Conversions'].sum().unstack().fillna(0)
heatmap_data = heatmap_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

fig3, ax3 = plt.subplots(figsize=(12, 5))
sns.heatmap(heatmap_data, cmap="YlOrBr", annot=True, fmt='.0f', ax=ax3)
ax3.set_title("Conversions Heatmap")
st.pyplot(fig3)