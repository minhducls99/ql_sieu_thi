"""
Module làm sạch dữ liệu (Data Cleaner)
=====================================
Chức năng chính:
- Xử lý giá trị thiếu (missing values)
- Xóa dữ liệu trùng lặp
- Xử lý outliers (giá trị bất thường)
- Mã hóa biến phân loại (categorical encoding)
- Chuẩn hóa dữ liệu (scaling)
- Xử lý cột ngày tháng
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning, preprocessing, and transformation.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a DataFrame.
        
        Args:
            df: Input DataFrame to clean
        """
        self.df = df.copy()
        self.cleaning_report = {}
        self.encoding_maps = {}
    
    # ==================== MISSING VALUES ====================
    
    def handle_missing_values(self, 
                              numeric_strategy: str = 'median',
                              categorical_strategy: str = 'mode',
                              threshold: float = 0.3) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            numeric_strategy: Strategy for numeric columns (mean, median, mode)
            categorical_strategy: Strategy for categorical columns (mode, constant)
            threshold: Drop columns with more than threshold proportion missing
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Handling missing values...")
        
        initial_missing = self.df.isnull().sum().sum()
        
        # Drop columns with too many missing values
        missing_ratio = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
            self.df = self.df.drop(columns=cols_to_drop)
            self.cleaning_report['dropped_columns'] = cols_to_drop
        
        # Handle numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                if numeric_strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif numeric_strategy == 'median':
                    fill_value = self.df[col].median()
                else:
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                
                self.df[col] = self.df[col].fillna(fill_value)
                logger.info(f"Filled {col} missing with {fill_value}")
        
        # Handle categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                if categorical_strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                else:
                    fill_value = 'Unknown'
                
                self.df[col] = self.df[col].fillna(fill_value)
                logger.info(f"Filled {col} missing with '{fill_value}'")
        
        final_missing = self.df.isnull().sum().sum()
        self.cleaning_report['missing_handled'] = initial_missing - final_missing
        
        return self.df
    
    # ==================== DUPLICATES ====================
    
    def handle_duplicates(self, subset: Optional[List[str]] = None, 
                         keep: str = 'first') -> Tuple[pd.DataFrame, int]:
        """
        Remove duplicate rows.
        
        Args:
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep (first, last, False)
            
        Returns:
            Tuple of (cleaned DataFrame, number of duplicates removed)
        """
        logger.info("Handling duplicates...")
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        duplicates_removed = initial_count - len(self.df)
        
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        
        return self.df, duplicates_removed
    
    # ==================== OUTLIERS ====================
    
    def handle_outliers_iqr(self, columns: List[str], 
                            threshold: float = 1.5,
                            method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers using IQR method.
        
        Args:
            columns: Columns to check for outliers
            threshold: IQR multiplier (default 1.5)
            method: How to handle outliers (clip, remove, NaN)
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Handling outliers with IQR method (threshold={threshold})...")
        
        outliers_info = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            if method == 'remove':
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                outliers_info[col] = (~mask).sum()
                self.df = self.df[mask]
            elif method == 'clip':
                outliers_info[col] = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
            elif method == 'NaN':
                outliers_info[col] = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                self.df.loc[(self.df[col] < lower_bound) | (self.df[col] > upper_bound), col] = np.nan
        
        self.cleaning_report['outliers_handled'] = outliers_info
        logger.info(f"Outliers handled: {outliers_info}")
        
        return self.df
    
    def handle_outliers_zscore(self, columns: List[str], 
                               threshold: float = 3,
                               method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers using Z-score method.
        
        Args:
            columns: Columns to check for outliers
            threshold: Z-score threshold
            method: How to handle outliers (clip, remove, NaN)
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Handling outliers with Z-score method (threshold={threshold})...")
        
        outliers_info = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            z_scores = np.abs(stats.zscore(self.df[col].fillna(0)))
            outliers = z_scores > threshold
            outliers_info[col] = outliers.sum()
            
            if method == 'remove':
                self.df = self.df[~outliers]
            elif method == 'clip':
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.df[col] = self.df[col].clip(mean - threshold*std, mean + threshold*std)
            elif method == 'NaN':
                self.df.loc[outliers, col] = np.nan
        
        self.cleaning_report['outliers_zscore'] = outliers_info
        
        return self.df
    
    # ==================== ENCODING ====================
    
    def encode_categorical(self, columns: List[str], 
                           method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            columns: Columns to encode
            method: Encoding method (label, onehot, ordinal)
            
        Returns:
            DataFrame with encoded columns
        """
        logger.info(f"Encoding categorical columns using {method}...")
        
        if method == 'label':
            for col in columns:
                if col in self.df.columns:
                    self.df[col + '_encoded'] = pd.factorize(self.df[col])[0]
                    self.encoding_maps[col] = dict(zip(self.df[col], self.df[col + '_encoded']))
                    logger.info(f"Label encoded {col}: {len(self.encoding_maps[col])} unique values")
        
        elif method == 'onehot':
            for col in columns:
                if col in self.df.columns:
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    logger.info(f"One-hot encoded {col}: {len(dummies.columns)} columns")
        
        return self.df
    
    def encode_frequency(self, columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical columns by frequency.
        
        Args:
            columns: Columns to encode
            
        Returns:
            DataFrame with frequency encoded columns
        """
        logger.info("Encoding by frequency...")
        
        for col in columns:
            if col in self.df.columns:
                freq_map = self.df[col].value_counts(normalize=True).to_dict()
                self.df[col + '_freq'] = self.df[col].map(freq_map)
                logger.info(f"Frequency encoded {col}")
        
        return self.df
    
    # ==================== SCALING ====================
    
    def scale_numeric(self, columns: List[str], 
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numeric columns.
        
        Args:
            columns: Columns to scale
            method: Scaling method (standard, minmax, robust)
            
        Returns:
            DataFrame with scaled columns
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        logger.info(f"Scaling numeric columns using {method}...")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        for col in columns:
            if col in self.df.columns:
                self.df[col + '_scaled'] = scaler.fit_transform(self.df[[col]])
                logger.info(f"Scaled {col}")
        
        return self.df
    
    # ==================== DATE PROCESSING ====================
    
    def process_dates(self, date_columns: List[str]) -> pd.DataFrame:
        """
        Process date columns - extract features.
        
        Args:
            date_columns: Columns containing dates
            
        Returns:
            DataFrame with date features
        """
        logger.info("Processing date columns...")
        
        for col in date_columns:
            if col in self.df.columns:
                # Convert to datetime if needed
                if self.df[col].dtype == 'object':
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                
                # Extract date features
                self.df[col + '_year'] = self.df[col].dt.year
                self.df[col + '_month'] = self.df[col].dt.month
                self.df[col + '_quarter'] = self.df[col].dt.quarter
                self.df[col + '_day'] = self.df[col].dt.day
                self.df[col + '_dayofweek'] = self.df[col].dt.dayofweek
                self.df[col + '_is_weekend'] = self.df[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                logger.info(f"Extracted date features from {col}")
        
        return self.df
    
    # ==================== DATA TYPE CONVERSION ====================
    
    def convert_types(self, type_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Convert column data types.
        
        Args:
            type_mapping: Dictionary mapping column names to desired types
            
        Returns:
            DataFrame with converted types
        """
        logger.info("Converting data types...")
        
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                if dtype == 'datetime':
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                elif dtype == 'numeric':
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                elif dtype == 'category':
                    self.df[col] = self.df[col].astype('category')
                else:
                    self.df[col] = self.df[col].astype(dtype)
                logger.info(f"Converted {col} to {dtype}")
        
        return self.df
    
    # ==================== REPORT ====================
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Get a summary report of all cleaning operations.
        
        Returns:
            Dictionary with cleaning report
        """
        return {
            'original_shape': self.cleaning_report.get('original_shape'),
            'final_shape': self.df.shape,
            'duplicates_removed': self.cleaning_report.get('duplicates_removed', 0),
            'missing_handled': self.cleaning_report.get('missing_handled', 0),
            'outliers_handled': self.cleaning_report.get('outliers_handled', {}),
            'encoding_maps': list(self.encoding_maps.keys())
        }
    
    # ==================== MAIN PREPROCESSING PIPELINE ====================
    
    def full_preprocessing_pipeline(self, 
                                    numeric_cols: List[str],
                                    categorical_cols: List[str],
                                    date_cols: List[str],
                                    outlier_cols: List[str]) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline.
        
        Args:
            numeric_cols: Numeric columns
            categorical_cols: Categorical columns
            date_cols: Date columns
            outlier_cols: Columns to check for outliers
            
        Returns:
            Fully preprocessed DataFrame
        """
        self.cleaning_report['original_shape'] = self.df.shape
        
        # 1. Handle missing values
        self.handle_missing_values(numeric_strategy='median', 
                                  categorical_strategy='mode')
        
        # 2. Handle duplicates
        self.handle_duplicates(subset=['Order ID', 'Product ID'])
        
        # 3. Handle outliers
        if outlier_cols:
            self.handle_outliers_iqr(outlier_cols, threshold=1.5, method='clip')
        
        # 4. Process dates
        if date_cols:
            self.process_dates(date_cols)
        
        # 5. Encode categorical
        self.encode_categorical(categorical_cols, method='label')
        
        # 6. Scale numeric
        self.scale_numeric(numeric_cols, method='standard')
        
        logger.info("Full preprocessing pipeline completed")
        
        return self.df


# Demo usage
if __name__ == "__main__":
    from src.data.loader import DataLoader
    
    loader = DataLoader()
    df = loader.generate_sample_data(n_orders=500)
    
    print("Original shape:", df.shape)
    
    cleaner = DataCleaner(df)
    cleaned = cleaner.full_preprocessing_pipeline(
        numeric_cols=['Sales', 'Quantity', 'Profit', 'Discount'],
        categorical_cols=['Category', 'Region', 'Segment', 'Ship Mode'],
        date_cols=['Order Date'],
        outlier_cols=['Sales', 'Profit']
    )
    
    print("\nCleaned shape:", cleaned.shape)
    print("\nCleaning report:", cleaner.get_cleaning_report())