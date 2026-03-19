"""
Model Evaluation Metrics Module
Provides comprehensive metrics for all model types
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, classification_report,
                            mean_absolute_error, mean_squared_error, r2_score,
                            silhouette_score, davies_bouldin_score, calinski_harabasz_score)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation across all model types.
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.evaluation_results = {}
    
    # ==================== CLASSIFICATION METRICS ====================
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               average: str = 'macro') -> Dict[str, float]:
        """
        Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            average: Averaging method for multi-class
            
        Returns:
            Dictionary of classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        
        # Add per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['precision_per_class'] = precision_per_class.tolist()
        metrics['recall_per_class'] = recall_per_class.tolist()
        metrics['f1_per_class'] = f1_per_class.tolist()
        
        # ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average=average)
            except:
                metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['confusion_matrix_normalized'] = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].tolist()
        
        return metrics
    
    # ==================== REGRESSION METRICS ====================
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # sMAPE
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        # R2
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'r2': r2
        }
    
    # ==================== CLUSTERING METRICS ====================
    
    def evaluate_clustering(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering model.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of clustering metrics
        """
        n_clusters = len(set(labels))
        
        if n_clusters < 2:
            return {'error': 'Need at least 2 clusters'}
        
        metrics = {
            'n_clusters': n_clusters,
            'silhouette': silhouette_score(features, labels),
            'davies_bouldin': davies_bouldin_score(features, labels),
            'calinski_harabasz': calinski_harabasz_score(features, labels)
        }
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        
        return metrics
    
    # ==================== TIME SERIES METRICS ====================
    
    def evaluate_forecasting(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate time series forecasting model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of forecasting metrics
        """
        metrics = self.evaluate_regression(y_true, y_pred)
        
        # Additional time series metrics
        
        # Direction accuracy (trend prediction)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['direction_accuracy'] = np.mean(true_direction == pred_direction)
        
        return metrics
    
    # ==================== COMPARISON ====================
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]], 
                      metric: str = 'f1_score',
                      sort_descending: bool = True) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results_dict: Dictionary of model results
            metric: Metric to compare by
            sort_descending: Sort in descending order
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for model_name, metrics in results_dict.items():
            row = {'Model': model_name}
            
            # Extract metric value
            if metric in metrics:
                row[metric] = metrics[metric]
            elif metric in str(metrics):
                # Try to find in nested dict
                for k, v in metrics.items():
                    if isinstance(v, dict) and metric in v:
                        row[metric] = v[metric]
            
            # Add all available metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k != metric:
                    row[k] = v
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=not sort_descending)
        
        return df
    
    # ==================== VISUALIZATION ====================
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str],
                            save_path: Optional[str] = None,
                            normalize: bool = False):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            save_path: Path to save figure
            normalize: Whether to normalize
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_metric_comparison(self, results_dict: Dict[str, Dict[str, Any]],
                              metrics: List[str],
                              save_path: Optional[str] = None):
        """
        Plot metric comparison across models.
        
        Args:
            results_dict: Dictionary of model results
            metrics: List of metrics to plot
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = []
            names = []
            
            for model_name, result in results_dict.items():
                if metric in result:
                    values.append(result[metric])
                    names.append(model_name)
            
            if values:
                axes[i].barh(names, values, color='steelblue')
                axes[i].set_xlabel(metric)
                axes[i].set_title(f'{metric} Comparison')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Metric comparison saved to {save_path}")
        
        plt.close()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None):
        """
        Plot residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # Residuals distribution
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residual Distribution')
        
        # Actual vs Predicted
        axes[1, 0].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].set_title('Actual vs Predicted')
        
        # Residuals over time/index
        axes[1, 1].plot(residuals, marker='o', linestyle='-', alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Over Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
        
        plt.close()
    
    # ==================== SUMMARY ====================
    
    def generate_summary(self, model_name: str, metrics: Dict[str, Any]) -> str:
        """
        Generate text summary of model performance.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            
        Returns:
            Summary string
        """
        summary = f"\n{'='*50}\n"
        summary += f"Model: {model_name}\n"
        summary += f"{'='*50}\n"
        
        for key, value in metrics.items():
            if isinstance(value, float):
                summary += f"{key}: {value:.4f}\n"
            elif isinstance(value, list) and len(value) <= 5:
                summary += f"{key}: {value}\n"
            elif isinstance(value, dict):
                summary += f"{key}:\n"
                for k, v in value.items():
                    summary += f"  {k}: {v}\n"
        
        summary += f"{'='*50}\n"
        
        return summary


# Demo usage
if __name__ == "__main__":
    # Test classification metrics
    evaluator = ModelEvaluator()
    
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1])
    y_pred_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.3, 0.6]
    ])
    
    cls_metrics = evaluator.evaluate_classification(y_true, y_pred, y_pred_proba)
    print("\nClassification Metrics:")
    print(evaluator.generate_summary("Test Model", cls_metrics))
    
    # Test regression metrics
    y_true_reg = np.array([1, 2, 3, 4, 5])
    y_pred_reg = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    reg_metrics = evaluator.evaluate_regression(y_true_reg, y_pred_reg)
    print("\nRegression Metrics:")
    print(evaluator.generate_summary("Test Model", reg_metrics))