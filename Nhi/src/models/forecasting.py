"""
Module dự báo chuỗi thời gian (Time Series Forecasting)
======================================================
Các mô hình dự báo được sử dụng:
- ARIMA: AutoRegressive Integrated Moving Average
- Holt-Winters: Dự báo với xu hướng và tính thời vụ
- Naive: Dự báo đơn giản - lấy giá trị cuối cùng
- Moving Average: Trung bình động

Các chỉ số đánh giá:
- MAE (Mean Absolute Error): Sai số tuyệt đối trung bình
- RMSE (Root Mean Squared Error): Căn bậc hai của bình phương sai số trung bình
- sMAPE (symmetric Mean Absolute Percentage Error): Phần trăm sai số tuyệt đối
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tắt các cảnh báo để output gọn hơn
warnings.filterwarnings('ignore')


class TimeSeriesModel:
    """
    Lớp mô hình dự báo chuỗi thời gian cho dự báo doanh số
    Sử dụng ARIMA, Holt-Winters, và các phương pháp cơ bản
    """
    
    def __init__(self, freq: str = 'W', test_size: float = 0.2):
        """
        Khởi tạo TimeSeriesModel
        
        Args:
            freq: Tần suất dữ liệu (D=ngày, W=tuần, M=tháng)
            test_size: Tỷ lệ dữ liệu dùng để test
        """
        self.freq = freq
        self.test_size = test_size
        self.models = {}
        self.results = {}
        self.train_data = None
        self.test_data = None
        self.ts = None
    
    def prepare_time_series(self, df: pd.DataFrame, 
                           date_col: str = 'Order Date',
                           value_col: str = 'Sales',
                           agg_func: str = 'sum') -> pd.Series:
        """
        Chuẩn bị dữ liệu chuỗi thời gian
        
        Args:
            df: DataFrame đầu vào
            date_col: Tên cột ngày tháng
            value_col: Cột giá trị cần dự báo
            agg_func: Hàm tổng hợp (sum, mean, ...)
            
        Returns:
            Chuỗi thời gian dạng pandas Series
        """
        logger.info(f"Preparing time series: {value_col} by {date_col}")
        
        # Đảm bảo cột ngày là datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Aggregate by time period
        ts = df.groupby(pd.Grouper(key=date_col, freq=self.freq))[value_col].agg(agg_func)
        ts = ts.fillna(0)
        
        # Remove leading/trailing zeros
        ts = ts[(ts != 0) | (ts.shift(1) != 0)]
        
        self.ts = ts
        logger.info(f"Time series shape: {len(ts)} periods")
        
        return ts
    
    def train_test_split(self, n_test: Optional[int] = None) -> Tuple:
        """
        Split time series into train and test.
        
        Args:
            n_test: Number of periods for test set
            
        Returns:
            Tuple of (train, test)
        """
        if self.ts is None:
            raise ValueError("No time series. Call prepare_time_series first.")
        
        if n_test is None:
            n_test = int(len(self.ts) * self.test_size)
        
        self.train_data = self.ts[:-n_test]
        self.test_data = self.ts[-n_test:]
        
        logger.info(f"Train: {len(self.train_data)}, Test: {len(self.test_data)}")
        
        return self.train_data, self.test_data
    
    def baseline_naive(self) -> Dict[str, Any]:
        """
        Naive baseline: use last value.
        
        Returns:
            Dictionary with predictions and metrics
        """
        logger.info("Running Naive baseline...")
        
        predictions = np.full(len(self.test_data), self.train_data.iloc[-1])
        
        metrics = self._calculate_metrics(self.test_data.values, predictions)
        
        self.models['Naive'] = {'predictions': predictions}
        self.results['Naive'] = metrics
        
        logger.info(f"Naive - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
        
        return {'predictions': predictions, 'metrics': metrics}
    
    def baseline_moving_average(self, window: int = 4) -> Dict[str, Any]:
        """
        Moving average baseline.
        
        Args:
            window: Window size
            
        Returns:
            Dictionary with predictions and metrics
        """
        logger.info(f"Running Moving Average (window={window})...")
        
        # For test, use last MA from train
        ma_value = self.train_data.iloc[-window:].mean()
        predictions = np.full(len(self.test_data), ma_value)
        
        metrics = self._calculate_metrics(self.test_data.values, predictions)
        
        self.models[f'MA_{window}'] = {'predictions': predictions}
        self.results[f'MA_{window}'] = metrics
        
        logger.info(f"MA-{window} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
        
        return {'predictions': predictions, 'metrics': metrics}
    
    def fit_arima(self, order: Tuple = (1, 1, 1), 
                  seasonal_order: Tuple = (1, 1, 1, 12)) -> Dict[str, Any]:
        """
        Fit ARIMA model.
        
        Args:
            order: (p, d, q) order
            seasonal_order: (P, D, Q, s) seasonal order
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Fitting ARIMA{order}...")
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Fit ARIMA
            model = ARIMA(self.train_data, order=order)
            model_fit = model.fit()
            
            # Predictions
            predictions = model_fit.forecast(steps=len(self.test_data))
            
            metrics = self._calculate_metrics(self.test_data.values, predictions.values)
            
            self.models['ARIMA'] = {
                'model': model_fit,
                'order': order,
                'predictions': predictions.values
            }
            self.results['ARIMA'] = metrics
            
            logger.info(f"ARIMA - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
            
            return {'model': model_fit, 'predictions': predictions.values, 'metrics': metrics}
            
        except Exception as e:
            logger.error(f"ARIMA failed: {str(e)}")
            return {'error': str(e)}
    
    def fit_holt_winters(self, seasonal: str = 'add', 
                        seasonal_periods: int = 12) -> Dict[str, Any]:
        """
        Fit Holt-Winters Exponential Smoothing.
        
        Args:
            seasonal: Seasonal component ('add' or 'mul')
            seasonal_periods: Number of seasonal periods
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Fitting Holt-Winters (seasonal={seasonal})...")
        
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            model = ExponentialSmoothing(
                self.train_data,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                trend='add',
                damped_trend=True
            )
            
            model_fit = model.fit()
            
            # Predictions
            predictions = model_fit.forecast(steps=len(self.test_data))
            
            metrics = self._calculate_metrics(self.test_data.values, predictions.values)
            
            self.models['HoltWinters'] = {
                'model': model_fit,
                'predictions': predictions.values
            }
            self.results['HoltWinters'] = metrics
            
            logger.info(f"HoltWinters - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
            
            return {'model': model_fit, 'predictions': predictions.values, 'metrics': metrics}
            
        except Exception as e:
            logger.error(f"Holt-Winters failed: {str(e)}")
            return {'error': str(e)}
    
    def fit_prophet(self, yearly_seasonality: bool = True,
                   weekly_seasonality: bool = True) -> Dict[str, Any]:
        """
        Fit Facebook Prophet model.
        
        Args:
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Fitting Prophet...")
        
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': self.train_data.index,
                'y': self.train_data.values
            })
            
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=False
            )
            
            model.fit(prophet_df)
            
            # Future dataframe
            future = pd.DataFrame({'ds': self.test_data.index})
            forecast = model.predict(future)
            
            predictions = forecast['yhat'].values
            
            metrics = self._calculate_metrics(self.test_data.values, predictions)
            
            self.models['Prophet'] = {
                'model': model,
                'predictions': predictions
            }
            self.results['Prophet'] = metrics
            
            logger.info(f"Prophet - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
            
            return {'model': model, 'predictions': predictions, 'metrics': metrics}
            
        except ImportError:
            logger.warning("Prophet not installed. Skipping...")
            return {'error': 'Prophet not available'}
        except Exception as e:
            logger.error(f"Prophet failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate forecasting metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # sMAPE
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'smape': smape,
            'mape': mape
        }
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('mae')
        
        return df
    
    def analyze_residuals(self, model_name: str = 'ARIMA') -> Dict[str, Any]:
        """
        Analyze model residuals.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with residual analysis
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        predictions = self.models[model_name]['predictions']
        residuals = self.test_data.values - predictions
        
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'residuals': residuals
        }
    
    def plot_forecast(self, save_path: Optional[str] = None):
        """
        Plot forecast results.
        
        Args:
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 6))
        
        # Plot train data
        plt.plot(self.train_data.index, self.train_data.values, 
                label='Train', color='blue', alpha=0.7)
        
        # Plot test data
        plt.plot(self.test_data.index, self.test_data.values, 
                label='Actual', color='green', linewidth=2)
        
        # Plot predictions
        colors = ['red', 'orange', 'purple', 'brown']
        for i, (model_name, model_data) in enumerate(self.models.items()):
            if 'predictions' in model_data:
                plt.plot(self.test_data.index, model_data['predictions'],
                        label=model_name, color=colors[i % len(colors)], 
                        linestyle='--', linewidth=1.5)
        
        plt.title('Sales Forecast Comparison')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")
        
        plt.close()
    
    def plot_residuals(self, model_name: str = 'ARIMA', 
                      save_path: Optional[str] = None):
        """
        Plot residual analysis.
        
        Args:
            model_name: Name of the model
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        residuals = self.analyze_residuals(model_name)['residuals']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Residuals over time
        axes[0, 0].plot(self.test_data.index, residuals)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title(f'{model_name} - Residuals')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residual')
        
        # Histogram
        axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        
        # ACF
        plot_acf(residuals, ax=axes[1, 0], lags=10)
        axes[1, 0].set_title('ACF of Residuals')
        
        # PACF
        plot_pacf(residuals, ax=axes[1, 1], lags=10)
        axes[1, 1].set_title('PACF of Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def get_seasonality(self) -> Dict[str, Any]:
        """
        Analyze seasonality in the data.
        
        Returns:
            Dictionary with seasonality information
        """
        if self.ts is None:
            raise ValueError("No time series prepared")
        
        # Monthly seasonality
        monthly = self.ts.groupby(self.ts.index.month).mean()
        
        # Day of week seasonality
        dow = self.ts.groupby(self.ts.index.dayofweek).mean()
        
        return {
            'monthly_pattern': monthly.to_dict(),
            'day_of_week_pattern': dow.to_dict(),
            'overall_mean': self.ts.mean(),
            'overall_std': self.ts.std()
        }


# Demo usage
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.cleaner import DataCleaner
    
    # Load and process data
    loader = DataLoader()
    df = loader.generate_sample_data(n_orders=500)
    
    cleaner = DataCleaner(df)
    cleaned = cleaner.full_preprocessing_pipeline(
        numeric_cols=['Sales', 'Quantity', 'Profit', 'Discount'],
        categorical_cols=['Category', 'Region', 'Segment', 'Ship Mode'],
        date_cols=['Order Date'],
        outlier_cols=['Sales', 'Profit']
    )
    
    # Prepare time series
    ts_model = TimeSeriesModel(freq='W', test_size=0.2)
    ts = ts_model.prepare_time_series(cleaned, 'Order Date', 'Sales', 'sum')
    train, test = ts_model.train_test_split(n_test=12)
    
    # Run baselines
    ts_model.baseline_naive()
    ts_model.baseline_moving_average(window=4)
    
    # Run models
    ts_model.fit_arima(order=(1, 1, 1))
    ts_model.fit_holt_winters(seasonal='add', seasonal_periods=12)
    
    print("\nModel Comparison:")
    print(ts_model.compare_models())