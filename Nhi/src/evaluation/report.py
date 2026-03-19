"""
Report Generation Module
Creates comprehensive reports and visualizations for all analyses
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive reports and visualizations.
    """
    
    def __init__(self, output_dir: str = "outputs/"):
        """
        Initialize ReportGenerator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = output_dir
        self.insights = []
    
    # ==================== EDA REPORTS ====================
    
    def generate_eda_report(self, df: pd.DataFrame, 
                          save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate EDA report.
        
        Args:
            df: DataFrame to analyze
            save_path: Path to save report
            
        Returns:
            Dictionary with EDA statistics
        """
        logger.info("Generating EDA report...")
        
        # Basic statistics
        stats = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        # Numeric statistics
        numeric_stats = df.describe().T.to_dict()
        stats['numeric_statistics'] = numeric_stats
        
        # Categorical statistics
        cat_stats = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            cat_stats[col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        stats['categorical_statistics'] = cat_stats
        
        self._add_insight(f"Dataset has {stats['n_rows']} rows and {stats['n_columns']} columns")
        self._add_insight(f"Found {stats['duplicate_rows']} duplicate rows")
        
        if save_path:
            self._save_report(stats, save_path)
        
        return stats
    
    def plot_distribution(self, df: pd.DataFrame, columns: List[str],
                        save_path: Optional[str] = None):
        """
        Plot distribution of columns.
        
        Args:
            df: DataFrame
            columns: Columns to plot
            save_path: Path to save figure
        """
        n_cols = len(columns)
        fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(14, 4 * ((n_cols + 1) // 2)))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for i, col in enumerate(columns):
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'{col} Distribution')
                else:
                    df[col].value_counts().head(10).plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'{col} Distribution')
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {save_path}")
        
        plt.close()
    
    def plot_correlation(self, df: pd.DataFrame, 
                        save_path: Optional[str] = None):
        """
        Plot correlation matrix.
        
        Args:
            df: DataFrame
            save_path: Path to save figure
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        plt.figure(figsize=(12, 10))
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Correlation plot saved to {save_path}")
        
        plt.close()
    
    # ==================== MINING REPORTS ====================
    
    def generate_association_report(self, rules: pd.DataFrame,
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate association rules report.
        
        Args:
            rules: DataFrame with association rules
            save_path: Path to save report
            
        Returns:
            Dictionary with report
        """
        logger.info("Generating association rules report...")
        
        if len(rules) == 0:
            return {'error': 'No rules to report'}
        
        report = {
            'n_rules': len(rules),
            'avg_support': rules['support'].mean(),
            'avg_confidence': rules['confidence'].mean(),
            'avg_lift': rules['lift'].mean(),
            'top_rules': rules.nlargest(10, 'lift').to_dict('records')
        }
        
        # Strongest associations
        self._add_insight(f"Found {len(rules)} association rules with average lift of {report['avg_lift']:.2f}")
        
        # High confidence rules
        high_conf = rules[rules['confidence'] >= 0.7]
        if len(high_conf) > 0:
            self._add_insight(f"{len(high_conf)} rules have high confidence (>=70%)")
        
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def generate_cluster_report(self, profiles: pd.DataFrame,
                               descriptions: Dict[int, str],
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate clustering report.
        
        Args:
            profiles: Cluster profiles
            descriptions: Cluster descriptions
            save_path: Path to save report
            
        Returns:
            Dictionary with report
        """
        logger.info("Generating cluster report...")
        
        report = {
            'n_clusters': len(profiles),
            'cluster_sizes': profiles['Size'].to_dict(),
            'cluster_percentages': profiles['Percentage'].to_dict(),
            'descriptions': descriptions,
            'profiles': profiles.to_dict('records')
        }
        
        # Largest cluster
        largest = profiles.loc[profiles['Size'].idxmax()]
        self._add_insight(f"Largest cluster: {int(largest['Cluster'])} with {largest['Size']} customers ({largest['Percentage']}%)")
        
        # Smallest cluster
        smallest = profiles.loc[profiles['Size'].idxmin()]
        self._add_insight(f"Smallest cluster: {int(smallest['Cluster'])} with {smallest['Size']} customers ({smallest['Percentage']}%)")
        
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    # ==================== MODEL COMPARISON ====================
    
    def generate_model_comparison(self, results_dict: Dict[str, Dict],
                                 save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate model comparison table.
        
        Args:
            results_dict: Dictionary of model results
            save_path: Path to save CSV
            
        Returns:
            DataFrame with comparison
        """
        logger.info("Generating model comparison...")
        
        comparison = []
        
        for model_name, metrics in results_dict.items():
            row = {'Model': model_name}
            
            # Flatten metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = value
                elif isinstance(value, list) and len(value) == 1:
                    row[key] = value[0]
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Sort by primary metric
        sort_col = 'f1_score' if 'f1_score' in df.columns else 'mae' if 'mae' in df.columns else 'silhouette' if 'silhouette' in df.columns else None
        if sort_col and sort_col in df.columns:
            if sort_col in ['mae', 'rmse', 'davies_bouldin']:
                df = df.sort_values(sort_col, ascending=True)
            else:
                df = df.sort_values(sort_col, ascending=False)
        
        # Best model
        if sort_col:
            best_model = df.iloc[0]['Model']
            best_value = df.iloc[0][sort_col]
            self._add_insight(f"Best model: {best_model} with {sort_col} = {best_value:.4f}")
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Model comparison saved to {save_path}")
        
        return df
    
    # ==================== ACTIONABLE INSIGHTS ====================
    
    def add_insight(self, insight: str):
        """
        Add an actionable insight.
        
        Args:
            insight: Insight text
        """
        self._add_insight(insight)
    
    def _add_insight(self, insight: str):
        """Internal method to add insight."""
        self.insights.append(insight)
        logger.info(f"Insight: {insight}")
    
    def generate_actionable_insights(self, 
                                     rfm_profiles: Optional[pd.DataFrame] = None,
                                     rules: Optional[pd.DataFrame] = None,
                                     cluster_profiles: Optional[pd.DataFrame] = None,
                                     forecasting_results: Optional[Dict] = None) -> List[str]:
        """
        Generate actionable business insights.
        
        Args:
            rfm_profiles: RFM analysis profiles
            rules: Association rules
            cluster_profiles: Cluster profiles
            forecasting_results: Forecasting results
            
        Returns:
            List of actionable insights
        """
        insights = []
        
        # RFM-based insights
        if rfm_profiles is not None and 'Segment' in rfm_profiles.columns:
            segment_counts = rfm_profiles['Segment'].value_counts()
            
            # Champions
            if 'Champions' in segment_counts:
                pct = segment_counts['Champions'] / len(rfm_profiles) * 100
                insights.append(f"🔒 Champions: {pct:.1f}% of customers - Prioritize loyalty programs and exclusive offers")
            
            # At Risk
            if 'At Risk' in segment_counts:
                pct = segment_counts['At Risk'] / len(rfm_profiles) * 100
                insights.append(f"⚠️ At Risk: {pct:.1f}% of customers - Implement win-back campaigns immediately")
            
            # Lost
            if 'Lost' in segment_counts:
                pct = segment_counts['Lost'] / len(rfm_profiles) * 100
                insights.append(f"❌ Lost: {pct:.1f}% of customers - Consider reactivation surveys")
        
        # Association rules insights
        if rules is not None and len(rules) > 0:
            top_rule = rules.nlargest(1, 'lift').iloc[0]
            ant = ', '.join(list(top_rule['antecedents']))
            cons = ', '.join(list(top_rule['consequents']))
            insights.append(f"📦 Cross-sell opportunity: Recommend {cons} when customers buy {ant} (lift: {top_rule['lift']:.2f})")
        
        # Clustering insights
        if cluster_profiles is not None:
            # High-value cluster
            if 'Monetary' in cluster_profiles.columns:
                high_value = cluster_profiles.loc[cluster_profiles['Monetary'].idxmax()]
                insights.append(f"💰 High-value segment (Cluster {int(high_value['Cluster'])}): Focus premium services")
            
            # High-frequency cluster  
            if 'Frequency' in cluster_profiles.columns:
                high_freq = cluster_profiles.loc[cluster_profiles['Frequency'].idxmax()]
                insights.append(f"🔄 Frequent buyers (Cluster {int(high_freq['Cluster'])}): Implement subscription programs")
        
        # Forecasting insights
        if forecasting_results is not None:
            if 'best_model' in forecasting_results:
                model = forecasting_results['best_model']
                if 'mae' in forecasting_results:
                    insights.append(f"📈 Sales forecast model ({model}) has MAE of {forecasting_results['mae']:.2f}")
        
        # General insights
        insights.extend([
            "🎯 Recommendation: Focus marketing budget on Champions and Potential Loyalists",
            "💡 Recommendation: Use cross-sell rules to increase average order value",
            "📊 Recommendation: Regular monitoring of customer segments for early intervention"
        ])
        
        self.insights.extend(insights)
        
        return insights
    
    def save_insights(self, save_path: str):
        """
        Save insights to file.
        
        Args:
            save_path: Path to save insights
        """
        with open(save_path, 'w') as f:
            f.write("ACTIONABLE INSIGHTS\n")
            f.write("=" * 50 + "\n\n")
            for i, insight in enumerate(self.insights, 1):
                f.write(f"{i}. {insight}\n")
        
        logger.info(f"Insights saved to {save_path}")
    
    # ==================== UTILITY METHODS ====================
    
    def _save_report(self, report: Dict, save_path: str):
        """
        Save report to file.
        
        Args:
            report: Report dictionary
            save_path: Path to save
        """
        import json
        
        # Convert non-serializable objects
        for key, value in report.items():
            if isinstance(value, pd.DataFrame):
                report[key] = value.to_dict()
            elif hasattr(value, 'tolist'):
                report[key] = value.tolist()
        
        with open(save_path.replace('.csv', '.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {save_path}")
    
    def generate_final_report(self, all_results: Dict[str, Any],
                            save_path: str = "outputs/reports/final_report.txt"):
        """
        Generate comprehensive final report.
        
        Args:
            all_results: Dictionary with all analysis results
            save_path: Path to save report
        """
        logger.info("Generating final report...")
        
        report = []
        report.append("=" * 70)
        report.append("SUPERSTORE SALES DATA MINING - FINAL REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Executive Summary
        report.append("\n1. EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        if 'summary_stats' in all_results:
            stats = all_results['summary_stats']
            report.append(f"Total Orders: {stats.get('total_orders', 'N/A')}")
            report.append(f"Total Customers: {stats.get('total_customers', 'N/A')}")
            report.append(f"Total Sales: ${stats.get('total_sales', 0):,.2f}")
            report.append(f"Total Profit: ${stats.get('total_profit', 0):,.2f}")
        
        # Key Findings
        report.append("\n2. KEY FINDINGS")
        report.append("-" * 40)
        
        if 'insights' in all_results:
            for i, insight in enumerate(all_results['insights'][:10], 1):
                report.append(f"{i}. {insight}")
        
        # Model Results
        report.append("\n3. MODEL PERFORMANCE")
        report.append("-" * 40)
        
        if 'classification_results' in all_results:
            report.append("\nClassification Models:")
            for model, metrics in all_results['classification_results'].items():
                if isinstance(metrics, dict) and 'f1_score' in metrics:
                    report.append(f"  - {model}: F1={metrics['f1_score']:.4f}")
        
        if 'clustering_results' in all_results:
            report.append("\nClustering Models:")
            for model, metrics in all_results['clustering_results'].items():
                if isinstance(metrics, dict) and 'silhouette' in metrics:
                    report.append(f"  - {model}: Silhouette={metrics['silhouette']:.4f}")
        
        if 'forecasting_results' in all_results:
            report.append("\nForecasting Models:")
            for model, metrics in all_results['forecasting_results'].items():
                if isinstance(metrics, dict) and 'mae' in metrics:
                    report.append(f"  - {model}: MAE={metrics['mae']:.2f}")
        
        # Actionable Recommendations
        report.append("\n4. ACTIONABLE RECOMMENDATIONS")
        report.append("-" * 40)
        
        if 'actionable_insights' in all_results:
            for i, insight in enumerate(all_results['actionable_insights'], 1):
                report.append(f"{i}. {insight}")
        
        # Technical Details
        report.append("\n5. TECHNICAL DETAILS")
        report.append("-" * 40)
        report.append(f"Project Structure: Data Mining Pipeline")
        report.append(f"Algorithms Used:")
        report.append("  - Association: Apriori")
        report.append("  - Clustering: K-Means, HAC")
        report.append("  - Classification: Logistic Regression, Decision Tree, Random Forest")
        report.append("  - Forecasting: ARIMA, Holt-Winters")
        
        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        # Save report
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Final report saved to {save_path}")
        
        return '\n'.join(report)


import os


# Demo usage
if __name__ == "__main__":
    generator = ReportGenerator(output_dir="outputs/")
    
    # Add insights
    generator.add_insight("Found 5 customer segments with distinct behaviors")
    generator.add_insight("Top association rule: Chairs → Tables with lift 2.5")
    
    # Generate actionable insights
    actionable = generator.generate_actionable_insights()
    
    print("\nActionable Insights:")
    for insight in actionable:
        print(f"  - {insight}")
    
    print("\nAll Insights:")
    for insight in generator.insights:
        print(f"  - {insight}")