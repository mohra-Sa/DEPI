import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# --- Title of the Dashboard ---
st.title("🎓 Student Performance Analysis Dashboard")

# --- Data Loading ---
@st.cache_data
def load_data():
    file_path = 'f:/progect data engineer/Students_clean (1).csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df['month'] = df['date'].dt.to_period('M')
    return df

df = load_data()

# --- Display Raw Data (Optional) ---
st.header("Raw Student Data Preview")
st.dataframe(df.head())

# --- Layout: Create Columns ---
# We create two columns to place charts side-by-side
col1, col2 = st.columns(2)

# --- CHART 1: PERFORMANCE TRENDS (in the first column) ---
with col1:
    st.header("Monthly Performance Trends")
    performance_trends = df.groupby(['month', 'performance']).size().unstack(fill_value=0)
    if 'Low' not in performance_trends.columns:
        performance_trends = performance_trends[['High', 'Medium']]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    performance_trends.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Monthly Student Performance Trends')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Students')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig1)

# --- CHART 2: ATTENDANCE HEATMAP (in the second column) ---
with col2:
    st.header("Average Monthly Attendance Heatmap")
    attendance_pivot = df.groupby(['Subject', 'month'])['Attendance (%)'].mean().unstack()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(attendance_pivot, ax=ax2, annot=True, fmt=".1f", cmap='viridis')
    ax2.set_title('Average Monthly Attendance (%) per Subject')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Subject')
    plt.tight_layout()
    st.pyplot(fig2)