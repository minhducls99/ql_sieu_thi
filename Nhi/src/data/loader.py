"""
Module tải dữ liệu (Data Loader)
Chịu trách nhiệm tải và kiểm tra dữ liệu bán hàng Superstore
==============================================================
Các chức năng chính:
- Tải dữ liệu từ file CSV/Excel
- Tạo dữ liệu mẫu khi không có dữ liệu thực
- Kiểm tra và thống kê dữ liệu ban đầu
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

# Cấu hình logging để hiển thị thông tin khi chạy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Lớp xử lý việc tải và kiểm tra dữ liệu ban đầu
    =================================================
    Thuộc tính:
        data_path: Đường dẫn tới thư mục chứa dữ liệu
        df: DataFrame chứa dữ liệu đã tải
        data_info: Dictionary lưu thông tin về dữ liệu
    """
    
    def __init__(self, data_path: str = "data/raw/"):
        """
        Khởi tạo DataLoader
        
        Args:
            data_path: Đường dẫn tới thư mục dữ liệu
        """
        self.data_path = Path(data_path)
        self.df = None
        self.data_info = {}
    
    def load_from_csv(self, filename: str = "superstore.csv") -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_path / filename
        
        if not filepath.exists():
            # Try to find the file with different extensions
            for ext in ['*.csv', '*.xlsx', '*.xls']:
                files = list(self.data_path.glob(ext))
                if files:
                    filepath = files[0]
                    break
        
        logger.info(f"Loading data from {filepath}")
        
        if filepath.suffix == '.csv':
            self.df = pd.read_csv(filepath, encoding='utf-8')
        elif filepath.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def load_from_kaggle(self) -> pd.DataFrame:
        """
        Load data from Kaggle (requires kaggle credentials).
        Falls back to sample data generation if not available.
        """
        try:
            import kaggle
            # This would work if Kaggle API is configured
            # For now, we'll generate sample data
            raise NotImplementedError("Kaggle API not configured. Use sample data.")
        except:
            logger.warning("Kaggle not available, generating sample data")
            return self.generate_sample_data()
    
    def generate_sample_data(self, n_orders: int = 5000) -> pd.DataFrame:
        """
        Generate sample Superstore data for demonstration.
        This mimics the structure of the real Superstore dataset.
        
        Args:
            n_orders: Number of orders to generate
            
        Returns:
            DataFrame with sample data
        """
        np.random.seed(42)
        
        # Categories and products
        categories = ['Furniture', 'Office Supplies', 'Technology']
        subcategories = {
            'Furniture': ['Chairs', 'Tables', 'Furnishings', 'Bookcases'],
            'Office Supplies': ['Labels', 'Paper', 'Binders', 'Supplies', 'Storage'],
            'Technology': ['Phones', 'Accessories', 'Machines', 'Copiers']
        }
        regions = ['East', 'West', 'Central', 'South']
        segments = ['Consumer', 'Corporate', 'Home Office']
        ship_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
        
        # Generate order IDs and dates
        order_ids = [f'OD-{i:06d}' for i in range(1, n_orders + 1)]
        
        # Generate dates from 2019-2023
        start_date = pd.Timestamp('2019-01-01')
        dates = pd.date_range(start=start_date, periods=n_orders*3, freq='D')
        order_dates = np.random.choice(dates, n_orders)
        
        # Convert to datetime for proper handling
        order_dates = pd.Series(order_dates)
        
        # Customer info
        customer_names = [f'Customer-{i:04d}' for i in np.random.randint(1, 1000, n_orders)]
        
        # Generate rows per order
        data = []
        for i in range(n_orders):
            n_items = np.random.randint(2, 6)  # At least 2 items for basket analysis
            order_date = pd.Timestamp(order_dates.iloc[i])
            
            # Generate different items for this order
            items_in_order = []
            for _ in range(n_items):
                cat = np.random.choice(categories)
                subcat = np.random.choice(subcategories[cat])
                items_in_order.append((cat, subcat))
            
            # Add each item to data
            for cat, subcat in items_in_order:
                row = {
                    'Row ID': len(data) + 1,
                    'Order ID': order_ids[i],
                    'Order Date': order_date,
                    'Ship Date': order_date + pd.Timedelta(days=np.random.randint(1, 7)),
                    'Ship Mode': np.random.choice(ship_modes),
                    'Customer ID': customer_names[i],
                    'Customer Name': customer_names[i],
                    'Segment': np.random.choice(segments),
                    'Country': 'United States',
                    'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 
                                              'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego']),
                    'State': np.random.choice(['New York', 'California', 'Illinois', 'Texas', 
                                               'Arizona', 'Pennsylvania', 'Texas', 'California']),
                    'Postal Code': np.random.randint(10000, 99999),
                    'Region': np.random.choice(regions),
                    'Product ID': f'PROD-{np.random.randint(1000, 9999)}',
                    'Category': cat,
                    'Sub-Category': subcat,
                    'Product Name': f'{subcat} - Item {np.random.randint(1, 100)}',
                    'Sales': round(np.random.uniform(10, 1000), 2),
                    'Quantity': np.random.randint(1, 10),
                    'Discount': round(np.random.choice([0, 0.05, 0.1, 0.15, 0.2]), 2),
                    'Profit': round(np.random.uniform(-50, 200), 2)
                }
                data.append(row)
        
        self.df = pd.DataFrame(data)
        logger.info(f"Generated {len(self.df)} sample records")
        return self.df
    
    def inspect_data(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Perform initial data inspection.
        
        Returns:
            Dictionary containing data information
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_from_csv() first.")
        
        self.data_info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'head': self.df.head().to_dict(),
            'describe': self.df.describe().to_dict(),
            'missing': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
        
        if verbose:
            self._print_inspection()
        
        return self.data_info
    
    def _print_inspection(self):
        """Print data inspection summary."""
        print("\n" + "="*60)
        print("DATA INSPECTION SUMMARY")
        print("="*60)
        print(f"\nShape: {self.data_info['shape']}")
        print(f"\nColumns: {', '.join(self.data_info['columns'])}")
        print(f"\nDuplicate rows: {self.data_info['duplicates']}")
        print("\nMissing values:")
        for col, missing in self.data_info['missing'].items():
            if missing > 0:
                print(f"  - {col}: {missing}")
        print("\nData types:")
        for col, dtype in self.data_info['dtypes'].items():
            print(f"  - {col}: {dtype}")
        print("="*60 + "\n")
    
    def get_column_info(self) -> pd.DataFrame:
        """
        Get detailed information about each column.
        
        Returns:
            DataFrame with column information
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        info_df = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes.values,
            'Non-Null Count': self.df.count().values,
            'Null Count': self.df.isnull().sum().values,
            'Unique Values': self.df.nunique().values,
            'Sample Values': [str(self.df[col].dropna().iloc[:3].tolist()) for col in self.df.columns]
        })
        
        return info_df
    
    def save_processed(self, output_path: str = "data/processed/", filename: str = "cleaned_data.csv"):
        """
        Save processed data to file.
        
        Args:
            output_path: Path to save the file
            filename: Name of the output file
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        filepath = Path(output_path) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        return {
            'total_orders': self.df['Order ID'].nunique() if 'Order ID' in self.df.columns else len(self.df),
            'total_customers': self.df['Customer ID'].nunique() if 'Customer ID' in self.df.columns else 0,
            'total_sales': self.df['Sales'].sum() if 'Sales' in self.df.columns else 0,
            'total_profit': self.df['Profit'].sum() if 'Profit' in self.df.columns else 0,
            'date_range': {
                'min': str(self.df['Order Date'].min()) if 'Order Date' in self.df.columns else None,
                'max': str(self.df['Order Date'].max()) if 'Order Date' in self.df.columns else None
            }
        }


# Demo usage
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.generate_sample_data(n_orders=1000)
    loader.inspect_data()
    print("\nSummary stats:", loader.get_summary_stats())