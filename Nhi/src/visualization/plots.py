"""
Visualization Plots Module
Common plotting functions for EDA and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_distribution(df: pd.DataFrame, column: str, 
                     bins: int = 30, 
                     kde: bool = True,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of a numeric column.
    
    Args:
        df: DataFrame
        column: Column name
        bins: Number of bins
        kde: Whether to show KDE
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if df[column].dtype in ['int64', 'float64']:
        sns.histplot(df[column].dropna(), bins=bins, kde=kde, ax=ax)
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {column}')
        
        # Add statistics
        stats = df[column].describe()
        stats_text = f"Mean: {stats['mean']:.2f}\nMedian: {df[column].median():.2f}\nStd: {stats['std']:.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame, 
                           columns: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame
        columns: Columns to include (default: all numeric)
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if columns:
        df = df[columns]
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
               cmap='coolwarm', center=0, square=True,
               linewidths=0.5, ax=ax)
    
    ax.set_title('Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_time_series(df: pd.DataFrame, 
                     date_col: str,
                     value_col: str,
                     resample_freq: str = 'W',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot time series data.
    
    Args:
        df: DataFrame
        date_col: Date column name
        value_col: Value column name
        resample_freq: Resampling frequency
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    ts = df[value_col].resample(resample_freq).sum()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(ts.index, ts.values, linewidth=1.5, color='steelblue')
    ax.fill_between(ts.index, ts.values, alpha=0.3)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col)
    ax.set_title(f'{value_col} Over Time ({resample_freq})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_category_distribution(df: pd.DataFrame, 
                               column: str,
                               top_n: int = 10,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot categorical distribution.
    
    Args:
        df: DataFrame
        column: Column name
        top_n: Number of top categories to show
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = df[column].value_counts().head(top_n)
    
    sns.barplot(x=counts.values, y=counts.index, palette='viridis', ax=ax)
    
    ax.set_xlabel('Count')
    ax.set_ylabel(column)
    ax.set_title(f'{column} Distribution (Top {top_n})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rfm_segments(rfm_df: pd.DataFrame, 
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot RFM segment distribution.
    
    Args:
        rfm_df: DataFrame with RFM data
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RFM Score distribution
    axes[0].hist(rfm_df['RFM_Score'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('RFM Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('RFM Score Distribution')
    
    # Segment distribution
    segment_counts = rfm_df['Segment'].value_counts()
    axes[1].pie(segment_counts.values, labels=segment_counts.index, 
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Customer Segments')
    
    # RFM scatter
    scatter = axes[2].scatter(rfm_df['Recency'], rfm_df['Monetary'], 
                             c=rfm_df['Frequency'], cmap='viridis', 
                             alpha=0.6, s=20)
    axes[2].set_xlabel('Recency (days)')
    axes[2].set_ylabel('Monetary ($)')
    axes[2].set_title('RFM Analysis')
    plt.colorbar(scatter, ax=axes[2], label='Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cluster_profiles(profiles_df: pd.DataFrame,
                         feature_cols: List[str],
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot cluster profiles as radar/bar chart.
    
    Args:
        profiles_df: DataFrame with cluster profiles
        feature_cols: Feature columns to plot
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize features for comparison
    profile_features = profiles_df[feature_cols].copy()
    for col in feature_cols:
        max_val = profile_features[col].max()
        if max_val > 0:
            profile_features[col] = profile_features[col] / max_val
    
    profile_features['Cluster'] = profiles_df['Cluster']
    
    # Plot as grouped bar chart
    profile_features.set_index('Cluster').plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Cluster Profiles (Normalized)')
    ax.legend(title='Features', bbox_to_anchor=(1.05, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(results_df: pd.DataFrame,
                         metric: str,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot model comparison bar chart.
    
    Args:
        results_df: DataFrame with model results
        metric: Metric to compare
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if metric in results_df.columns:
        results_df = results_df.sort_values(metric, ascending=True)
        
        # Color by better/worse
        colors = ['green' if metric in ['silhouette', 'f1_score', 'accuracy', 'r2'] 
                  else 'red'] * len(results_df)
        
        sns.barplot(x=metric, y='Model', data=results_df, palette=colors, ax=ax)
        
        ax.set_xlabel(metric)
        ax.set_ylabel('Model')
        ax.set_title(f'Model Comparison by {metric}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_forecast_comparison(actual: pd.Series,
                           predictions: Dict[str, np.ndarray],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot forecast comparison.
    
    Args:
        actual: Actual time series
        predictions: Dictionary of predictions by model name
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot actual
    ax.plot(actual.index, actual.values, 'k-', linewidth=2, label='Actual')
    
    # Plot predictions
    colors = ['red', 'blue', 'green', 'orange']
    for i, (model_name, pred) in enumerate(predictions.items()):
        ax.plot(actual.index, pred, '--', color=colors[i % len(colors)], 
                linewidth=1.5, label=model_name)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Forecast Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(feature_importance_df: pd.DataFrame,
                           top_n: int = 15,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_importance_df: DataFrame with Feature and Importance columns
        top_n: Number of top features to show
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = feature_importance_df.head(top_n)
    
    sns.barplot(x='Importance', y='Feature', data=top_features, 
                palette='viridis', ax=ax)
    
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {top_n} Feature Importance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig