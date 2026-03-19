"""
Module xây dựng đặc trưng (Feature Builder)
============================================
Chức năng chính:
- Tạo đặc trưng RFM (Recency, Frequency, Monetary) để phân khúc khách hàng
- Tạo dữ liệu giỏ hàng (basket data) cho khai phá luật kết hợp (association rules)
- Tạo đặc trưng khách hàng (customer features)
- Tạo đặc trưng sản phẩm (product features)
- Tạo đặc trưng thời gian (time features) cho dự báo chuỗi thời gian
- Tạo đặc trưng lag (lag features) cho dự báo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Lớp xây dựng đặc trưng cho dataset Superstore
    Tạo RFM, basket data, và các đặc trưng phái sinh khác
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Khởi tạo FeatureBuilder
        
        Args:
            df: DataFrame đầu vào
        """
        self.df = df.copy()
        self.rfm_features = None  # Lưu trữ đặc trưng RFM
        self.basket_data = None   # Lưu trữ dữ liệu giỏ hàng
    
    # ==================== ĐẶC TRƯNG RFM ====================
    # RFM là phương pháp phân khúc khách hàng dựa trên 3 tiêu chí:
    # - Recency (Độ gần đây): Thời gian từ lần mua cuối cùng
    # - Frequency (Tần suất): Số lần mua hàng
    # - Monetary (Tiền tệ): Tổng số tiền đã chi tiêu
    
    def create_rfm_features(self, 
                           reference_date: Optional[datetime] = None,
                           recency_bins: int = 5,
                           frequency_bins: int = 5,
                           monetary_bins: int = 5) -> pd.DataFrame:
        """
        Tạo đặc trưng RFM (Recency, Frequency, Monetary) cho mỗi khách hàng
        
        Args:
            reference_date: Ngày tham chiếu để tính recency
            recency_bins: Số bins cho điểm recency
            frequency_bins: Số bins cho điểm frequency  
            monetary_bins: Số bins cho điểm monetary
            
        Returns:
            DataFrame với đặc trưng RFM cho mỗi khách hàng
        """
        logger.info("Creating RFM features...")
        
        # Kiểm tra cột Order Date tồn tại
        if 'Order Date' not in self.df.columns:
            raise ValueError("Order Date column required for RFM")
        
        # Chuyển đổi Order Date sang datetime nếu cần
        if self.df['Order Date'].dtype != 'datetime64[ns]':
            self.df['Order Date'] = pd.to_datetime(self.df['Order Date'], errors='coerce')
        
        # Đặt ngày tham chiếu (mặc định là ngày sau ngày đơn hàng cuối cùng)
        if reference_date is None:
            reference_date = self.df['Order Date'].max() + timedelta(days=1)
        
        # Tính toán các chỉ số RFM cho mỗi khách hàng
        # groupby('Customer ID') nhóm dữ liệu theo từng khách hàng
        rfm = self.df.groupby('Customer ID').agg({
            'Order Date': lambda x: (reference_date - x.max()).days,  # Recency: Số ngày từ lần mua cuối
            'Order ID': 'nunique',  # Frequency: Số đơn hàng duy nhất
            'Sales': 'sum',  # Monetary: Tổng doanh số
            'Profit': 'sum',  # Tổng lợi nhuận
            'Quantity': 'sum'  # Tổng số lượng sản phẩm
        }).reset_index()
        
        # Đặt tên cột
        rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary', 'Total_Profit', 'Total_Quantity']
        
        # Tạo điểm RFM sử dụng quintiles (5 phần bằng nhau)
        # pd.qcut chia dữ liệu thành các phần bằng nhau dựa trên percentiles
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=recency_bins, labels=False, duplicates='drop').astype(int) + 1
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=frequency_bins, labels=False, duplicates='drop').astype(int) + 1
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=monetary_bins, labels=False, duplicates='drop').astype(int) + 1
        
        # Tổng hợp điểm RFM (kết hợp 3 điểm thành 1)
        # Ví dụ: R=5, F=4, M=3 => RFM_Score = 5*100 + 4*10 + 3 = 543
        rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']
        
        # Phân khúc khách hàng dựa trên điểm RFM
        rfm['Segment'] = rfm.apply(self._get_rfm_segment, axis=1)
        
        # Thống kê
        logger.info(f"Created RFM features for {len(rfm)} customers")
        logger.info(f"Segment distribution:\n{rfm['Segment'].value_counts()}")
        
        self.rfm_features = rfm
        return rfm
    
    def _get_rfm_segment(self, row) -> str:
        """
        Phân khúc khách hàng dựa trên điểm RFM
        
        Args:
            row: Dòng dữ liệu RFM
            
        Returns:
            Tên phân khúc khách hàng
        """
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Champions: Điểm cao trên cả 3 tiêu chí - KHÁCH HÀNG VÀNG
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        # Loyal Customers: Tần suất cao - KHÁCH HÀNG TRUNG THÀNH
        elif f >= 4:
            return 'Loyal Customers'
        # Potential Loyalists: Gần đây và tần suất khá - KHÁCH HÀNG TIỀM NĂNG
        elif r >= 3 and f >= 3:
            return 'Potential Loyalists'
        # At Risk: Mua lâu nhưng từng mua thường xuyên - KHÁCH HÀNG CÓ NGUY CƠ MẤT
        elif r <= 2 and f >= 3:
            return 'At Risk'
        # Lost: Không mua lâu và ít tần suất - KHÁCH HÀNG ĐÃ MẤT
        elif r <= 2 and f <= 2:
            return 'Lost'
        # New Customers: Mới mua gần đây nhưng ít tần suất - KHÁCH HÀNG MỚI
        elif r >= 4 and f <= 2:
            return 'New Customers'
        # Others: Các trường hợp khác - KHÁCH HÀNG TRIỂN VỌNG
        else:
            return 'Promising'
    
    # ==================== BASKET DATA ====================
    
    def create_basket_data(self, 
                          min_items: int = 2,
                          min_order_value: float = 10) -> pd.DataFrame:
        """
        Create basket data (transactions) for association rule mining.
        
        Args:
            min_items: Minimum items in a basket
            min_order_value: Minimum order value
            
        Returns:
            DataFrame with one row per order containing list of items
        """
        logger.info("Creating basket data...")
        
        # Filter by order value
        order_values = self.df.groupby('Order ID')['Sales'].sum()
        valid_orders = order_values[order_values >= min_order_value].index
        
        basket = self.df[self.df['Order ID'].isin(valid_orders)].groupby('Order ID').apply(
            lambda x: list(set(x['Sub-Category'].tolist()))
        ).reset_index()
        
        basket.columns = ['Order ID', 'Items']
        basket['Item_Count'] = basket['Items'].apply(len)
        
        # Filter by minimum items
        basket = basket[basket['Item_Count'] >= min_items]
        
        logger.info(f"Created {len(basket)} baskets with min {min_items} items")
        
        self.basket_data = basket
        return basket
    
    def create_transaction_matrix(self) -> pd.DataFrame:
        """
        Create transaction matrix (one-hot encoded) for Apriori.
        
        Returns:
            DataFrame with one-hot encoded transactions
        """
        if self.basket_data is None:
            self.create_basket_data()
        
        # Get all unique items
        all_items = set()
        for items in self.basket_data['Items']:
            all_items.update(items)
        
        # Create one-hot encoding
        transactions = pd.DataFrame(
            [[item in items for item in all_items] for items in self.basket_data['Items']],
            columns=list(all_items),
            index=self.basket_data['Order ID']
        )
        
        logger.info(f"Created transaction matrix: {transactions.shape}")
        
        return transactions
    
    # ==================== CUSTOMER FEATURES ====================
    
    def create_customer_features(self) -> pd.DataFrame:
        """
        Create additional customer-level features.
        
        Returns:
            DataFrame with customer features
        """
        logger.info("Creating customer features...")
        
        # Aggregate by customer
        customer_features = self.df.groupby('Customer ID').agg({
            'Order ID': 'nunique',
            'Order Date': ['min', 'max'],
            'Sales': ['sum', 'mean', 'std', 'min', 'max'],
            'Profit': ['sum', 'mean'],
            'Quantity': ['sum', 'mean'],
            'Discount': ['mean', 'max'],
            'Category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
            'Region': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        })
        
        # Flatten column names
        customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns]
        customer_features = customer_features.reset_index()
        
        # Rename columns
        customer_features = customer_features.rename(columns={
            'Order ID_nunique': 'Total_Orders',
            'Order Date_min': 'First_Order_Date',
            'Order Date_max': 'Last_Order_Date',
            'Sales_sum': 'Total_Sales',
            'Sales_mean': 'Avg_Order_Value',
            'Sales_std': 'Sales_Variance',
            'Sales_min': 'Min_Order_Value',
            'Sales_max': 'Max_Order_Value',
            'Profit_sum': 'Total_Profit',
            'Profit_mean': 'Avg_Profit',
            'Quantity_sum': 'Total_Quantity',
            'Quantity_mean': 'Avg_Quantity',
            'Discount_mean': 'Avg_Discount',
            'Discount_max': 'Max_Discount',
            'Category_mode': 'Preferred_Category',
            'Region_mode': 'Preferred_Region'
        })
        
        # Calculate additional metrics
        customer_features['Order_Frequency'] = customer_features['Total_Orders'] / (
            (customer_features['Last_Order_Date'] - customer_features['First_Order_Date']).dt.days + 1
        ) * 30  # Orders per month
        
        customer_features['Profit_Margin'] = customer_features['Total_Profit'] / customer_features['Total_Sales']
        
        logger.info(f"Created {len(customer_features)} customer profiles")
        
        return customer_features
    
    # ==================== PRODUCT FEATURES ====================
    
    def create_product_features(self) -> pd.DataFrame:
        """
        Create product-level features.
        
        Returns:
            DataFrame with product features
        """
        logger.info("Creating product features...")
        
        product_features = self.df.groupby(['Product ID', 'Product Name']).agg({
            'Order ID': 'nunique',
            'Customer ID': 'nunique',
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Quantity': ['sum', 'mean'],
            'Discount': 'mean'
        }).reset_index()
        
        product_features.columns = ['Product ID', 'Product Name', 'Total_Orders', 
                                    'Unique_Customers', 'Total_Sales', 'Avg_Sales',
                                    'Total_Profit', 'Avg_Profit', 'Total_Quantity',
                                    'Avg_Quantity', 'Avg_Discount']
        
        product_features['Profit_Margin'] = (
            product_features['Total_Profit'] / product_features['Total_Sales']
        )
        
        logger.info(f"Created {len(product_features)} product profiles")
        
        return product_features
    
    # ==================== TIME FEATURES ====================
    
    def create_time_features(self, date_col: str = 'Order Date') -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            date_col: Name of the date column
            
        Returns:
            DataFrame with time features
        """
        logger.info("Creating time features...")
        
        if date_col not in self.df.columns:
            raise ValueError(f"Column {date_col} not found")
        
        df = self.df.copy()
        
        # Date features
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Quarter'] = df[date_col].dt.quarter
        df['Day'] = df[date_col].dt.day
        df['DayOfWeek'] = df[date_col].dt.dayofweek
        df['WeekOfYear'] = df[date_col].dt.isocalendar().week
        
        # Cyclical encoding for month and day of week
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Is weekend
        df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Season
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Is holiday season (Nov-Dec)
        df['Is_Holiday_Season'] = df['Month'].isin([11, 12]).astype(int)
        
        self.df = df
        logger.info("Time features created")
        
        return df
    
    # ==================== LAG FEATURES ====================
    
    def create_lag_features(self, 
                           group_col: str = 'Customer ID',
                           target_col: str = 'Sales',
                           lags: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
        """
        Create lag features for time series forecasting.
        
        Args:
            group_col: Column to group by (e.g., Customer ID, Product ID)
            target_col: Column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        logger.info(f"Creating lag features for {target_col}...")
        
        df = self.df.sort_values(['Customer ID', 'Order Date'])
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
        
        # Rolling features
        df[f'{target_col}_rolling_mean_3'] = df.groupby(group_col)[target_col].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df[f'{target_col}_rolling_mean_6'] = df.groupby(group_col)[target_col].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        
        # Difference features
        df[f'{target_col}_diff'] = df.groupby(group_col)[target_col].diff()
        
        self.df = df
        logger.info("Lag features created")
        
        return df
    
    # ==================== COMBINE ALL ====================
    
    def get_all_features(self) -> Dict[str, pd.DataFrame]:
        """
        Get all created feature sets.
        
        Returns:
            Dictionary of feature DataFrames
        """
        return {
            'rfm': self.rfm_features,
            'basket': self.basket_data,
            'customers': self.create_customer_features(),
            'products': self.create_product_features(),
            'main': self.df
        }


# Demo usage
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.cleaner import DataCleaner
    
    loader = DataLoader()
    df = loader.generate_sample_data(n_orders=500)
    
    cleaner = DataCleaner(df)
    cleaned = cleaner.full_preprocessing_pipeline(
        numeric_cols=['Sales', 'Quantity', 'Profit', 'Discount'],
        categorical_cols=['Category', 'Region', 'Segment', 'Ship Mode'],
        date_cols=['Order Date'],
        outlier_cols=['Sales', 'Profit']
    )
    
    builder = FeatureBuilder(cleaned)
    
    # Test RFM
    rfm = builder.create_rfm_features()
    print("\nRFM features:")
    print(rfm.head())
    
    # Test Basket
    basket = builder.create_basket_data(min_items=2)
    print("\nBasket data:")
    print(basket.head())