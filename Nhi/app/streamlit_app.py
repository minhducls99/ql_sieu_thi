"""
Ứng dụng Dashboard Streamlit
============================
Dashboard tương tác để phân tích dữ liệu bán hàng Superstore

Các tính năng:
- Hiển thị dữ liệu và thống kê tổng quan
- Phân tích RFM và phân khúc khách hàng
- Khai phá luật kết hợp (Association Rules)
- Phân cụm khách hàng (Clustering)
- Dự báo doanh số (Sales Forecasting)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Thêm thư mục cha vào đường dẫn để import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import AssociationMiner
from src.mining.clustering import ClusterMiner
from sklearn.preprocessing import StandardScaler


# Cấu hình trang
st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# CSS tùy chỉnh để trang trí dashboard
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
    }
    .subtitle {
        font-size: 20px;
        color: #5D6D7E;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Tải và xử lý dữ liệu"""
    loader = DataLoader()
    df = loader.generate_sample_data(n_orders=2000)
    return df


@st.cache_data
def process_data(df):
    """Process data and create features"""
    cleaner = DataCleaner(df)
    df = cleaner.handle_missing_values()
    
    builder = FeatureBuilder(df)
    rfm = builder.create_rfm_features()
    basket = builder.create_basket_data(min_items=2)
    
    return rfm, basket


# Sidebar
st.sidebar.title("🛒 Superstore Analytics")
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "EDA", "Customer Segmentation", "Association Rules", "Forecasting", "About"]
)

# Main content
if page == "Home":
    st.markdown('<p class="title">Superstore Sales Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Data Mining Project - Customer Analysis & Forecasting</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Orders", f"{len(df):,}")
    
    with col2:
        st.metric("Total Customers", f"{df['Customer ID'].nunique():,}")
    
    with col3:
        st.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    
    with col4:
        st.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
    
    st.markdown("---")
    
    # Quick overview
    st.subheader("📊 Quick Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by category
        cat_sales = df.groupby('Category')['Sales'].sum()
        fig, ax = plt.subplots(figsize=(8, 6))
        cat_sales.plot(kind='bar', ax=ax, color=['#3498DB', '#E74C3C', '#2ECC71'])
        ax.set_title('Sales by Category')
        ax.set_ylabel('Sales ($)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Sales by region
        region_sales = df.groupby('Region')['Sales'].sum()
        fig, ax = plt.subplots(figsize=(8, 6))
        region_sales.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title('Sales by Region')
        ax.set_ylabel('')
        st.pyplot(fig)
    
    st.info("👈 Use the sidebar to navigate to different analyses!")


elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")
    
    df = load_data()
    
    # Data info
    st.subheader("Data Overview")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Columns
    st.subheader("Column Information")
    st.dataframe(pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Unique': df.nunique().values
    }))
    
    # Distribution
    st.subheader("Sales Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    df['Sales'].hist(bins=30, ax=ax, edgecolor='black')
    ax.set_xlabel('Sales ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sales')
    st.pyplot(fig)
    
    # Correlation
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)


elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation")
    
    with st.spinner("Loading RFM analysis..."):
        df = load_data()
        rfm, _ = process_data(df)
    
    # RFM scores
    st.subheader("RFM Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_recency = rfm['Recency'].mean()
        st.metric("Avg Recency (days)", f"{avg_recency:.0f}")
    
    with col2:
        avg_frequency = rfm['Frequency'].mean()
        st.metric("Avg Frequency", f"{avg_frequency:.1f}")
    
    with col3:
        avg_monetary = rfm['Monetary'].mean()
        st.metric("Avg Monetary ($)", f"${avg_monetary:,.0f}")
    
    # Segments
    st.subheader("Customer Segments")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    segment_counts = rfm['Segment'].value_counts()
    colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#9B59B6', '#1ABC9C', '#E67E22']
    ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
           colors=colors[:len(segment_counts)])
    ax.set_title('Customer Segments Distribution')
    st.pyplot(fig)
    
    # RFM Score distribution
    st.subheader("RFM Scores")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(rfm['R_Score'], bins=5, edgecolor='black')
    axes[0].set_title('Recency Score')
    
    axes[1].hist(rfm['F_Score'], bins=5, edgecolor='black')
    axes[1].set_title('Frequency Score')
    
    axes[2].hist(rfm['M_Score'], bins=5, edgecolor='black')
    axes[2].set_title('Monetary Score')
    
    st.pyplot(fig)
    
    # Segment details
    st.subheader("Segment Details")
    segment_stats = rfm.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Customer ID': 'count'
    }).round(2)
    segment_stats.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Count']
    st.dataframe(segment_stats)


elif page == "Association Rules":
    st.title("🔗 Association Rules")
    
    with st.spinner("Mining association rules..."):
        df = load_data()
        _, basket = process_data(df)
        
        transactions = basket['Items'].tolist()
        miner = AssociationMiner(min_support=0.02, min_confidence=0.3)
        result = miner.fit(transactions)
    
    # Parameters
    st.sidebar.subheader("Parameters")
    min_support = st.sidebar.slider("Min Support", 0.01, 0.1, 0.02)
    min_confidence = st.sidebar.slider("Min Confidence", 0.1, 0.9, 0.3)
    
    # Re-run with parameters
    miner = AssociationMiner(min_support=min_support, min_confidence=min_confidence)
    miner.fit(transactions)
    
    # Metrics
    metrics = miner.get_rule_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rules", metrics.get('total_rules', 0))
    
    with col2:
        st.metric("Avg Support", f"{metrics.get('avg_support', 0):.4f}")
    
    with col3:
        st.metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.2%}")
    
    with col4:
        st.metric("Avg Lift", f"{metrics.get('avg_lift', 0):.2f}")
    
    # Top rules
    st.subheader("Top Association Rules")
    top_rules = miner.get_top_rules(n=20, sort_by='lift')
    
    if len(top_rules) > 0:
        st.dataframe(top_rules, use_container_width=True)
    else:
        st.warning("No rules found with current parameters. Try lowering min_support or min_confidence.")
    
    # Insights
    st.subheader("💡 Business Insights")
    for insight in miner.generate_insights():
        st.write(f"• {insight}")


elif page == "Forecasting":
    st.title("📊 Sales Forecasting")
    
    st.info("Forecasting requires time series analysis. This feature is in development.")
    
    # Placeholder for forecasting
    st.subheader("Sales Trend Over Time")
    
    df = load_data()
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Month'] = df['Order Date'].dt.to_period('M')
    
    monthly = df.groupby('Month')['Sales'].sum()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly.plot(ax=ax, marker='o')
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales ($)')
    ax.set_title('Monthly Sales Trend')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("""
    **Forecasting Methods:**
    - **ARIMA**: Autoregressive Integrated Moving Average
    - **Holt-Winters**: Exponential Smoothing with trend and seasonality
    - **Prophet**: Facebook's forecasting tool
    """)


elif page == "About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## Superstore Sales Data Mining Project
    
    This is a comprehensive data mining project that analyzes supermarket sales data
    using various machine learning and data mining techniques.
    
    ### Techniques Used:
    
    1. **EDA & Preprocessing**
       - Data cleaning and missing value handling
       - Outlier detection and treatment
       - Feature engineering
    
    2. **Customer Segmentation (RFM)**
       - Recency, Frequency, Monetary analysis
       - K-Means clustering for customer groups
    
    3. **Association Rules**
       - Apriori algorithm for market basket analysis
       - Cross-sell and upsell recommendations
    
    4. **Classification**
       - Logistic Regression, Decision Tree, Random Forest
       - Customer segment prediction
    
    5. **Time Series Forecasting**
       - ARIMA, Holt-Winters
       - Sales prediction
    
    ### Tech Stack:
    - Python 3
    - Pandas, NumPy
    - Scikit-learn
    - MLxtend (Apriori)
    - Statsmodels
    - Streamlit
    """)
    
    st.markdown("---")
    st.markdown("**Project Structure:**")
    st.code("""
    DATA_MINING_PROJECT/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_preprocess_feature.ipynb
    │   ├── 03_mining_clustering.ipynb
    │   ├── 04_modeling.ipynb
    │   └── 05_evaluation_report.ipynb
    ├── src/
    │   ├── data/
    │   ├── features/
    │   ├── mining/
    │   ├── models/
    │   └── evaluation/
    ├── configs/
    │   └── params.yaml
    ├── scripts/
    │   └── run_pipeline.py
    └── outputs/
        ├── figures/
        ├── tables/
        └── reports/
    """)


# Run the app: streamlit run app/streamlit_app.py