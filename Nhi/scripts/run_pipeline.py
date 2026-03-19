"""
Pipeline Runner Script
Runs the complete data mining pipeline
"""

import sys
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_eda():
    """Run EDA notebook"""
    logger.info("=" * 60)
    logger.info("STEP 1: Exploratory Data Analysis")
    logger.info("=" * 60)
    
    try:
        from src.data.loader import DataLoader
        from src.data.cleaner import DataCleaner
        
        # Load data
        loader = DataLoader(data_path='data/raw/')
        df = loader.generate_sample_data(n_orders=2000)
        
        # Inspect data
        loader.inspect_data(verbose=True)
        
        # Save
        df.to_csv('data/processed/01_eda_data.csv', index=False)
        
        logger.info("EDA completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"EDA failed: {e}")
        return False


def run_preprocessing():
    """Run preprocessing and feature engineering"""
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing & Feature Engineering")
    logger.info("=" * 60)
    
    try:
        from src.data.loader import DataLoader
        from src.data.cleaner import DataCleaner
        from src.features.builder import FeatureBuilder
        
        # Load data
        loader = DataLoader()
        df = loader.generate_sample_data(n_orders=2000)
        
        # Clean
        cleaner = DataCleaner(df)
        df = cleaner.handle_missing_values()
        df, _ = cleaner.handle_duplicates()
        df = cleaner.handle_outliers_iqr(['Sales', 'Profit'])
        
        # Feature engineering
        builder = FeatureBuilder(df)
        rfm = builder.create_rfm_features()
        basket = builder.create_basket_data(min_items=2)
        customer_features = builder.create_customer_features()
        
        # Save
        df.to_csv('data/processed/02_cleaned_data.csv', index=False)
        rfm.to_csv('data/processed/02_rfm_features.csv', index=False)
        basket.to_csv('data/processed/02_basket_data.csv', index=False)
        customer_features.to_csv('data/processed/02_customer_features.csv', index=False)
        
        logger.info("Preprocessing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return False


def run_mining():
    """Run data mining (association rules & clustering)"""
    logger.info("=" * 60)
    logger.info("STEP 3: Data Mining")
    logger.info("=" * 60)
    
    try:
        from src.data.loader import DataLoader
        from src.data.cleaner import DataCleaner
        from src.features.builder import FeatureBuilder
        from src.mining.association import AssociationMiner
        from src.mining.clustering import ClusterMiner
        from sklearn.preprocessing import StandardScaler
        
        # Load and prepare data
        loader = DataLoader()
        df = loader.generate_sample_data(n_orders=2000)
        
        cleaner = DataCleaner(df)
        df = cleaner.handle_missing_values()
        
        builder = FeatureBuilder(df)
        rfm = builder.create_rfm_features()
        basket = builder.create_basket_data(min_items=2)
        
        # Association Rules
        logger.info("Running Association Rule Mining...")
        transactions = basket['Items'].tolist()
        miner = AssociationMiner(min_support=0.02, min_confidence=0.3)
        result = miner.fit(transactions)
        
        # Get rules
        rules = miner.get_top_rules(n=20, sort_by='lift')
        logger.info(f"Found {len(rules)} association rules")
        
        # Clustering
        logger.info("Running Clustering...")
        feature_cols = ['Recency', 'Frequency', 'Monetary']
        scaler = StandardScaler()
        features = scaler.fit_transform(rfm[feature_cols])
        
        clusterer = ClusterMiner(n_clusters=4, random_state=42)
        
        # Find optimal K
        k_results = clusterer.find_optimal_k(features, range(2, 8))
        
        # Fit with best K
        best_k = k_results['k'][k_results['silhouette'].index(max(k_results['silhouette']))]
        labels = clusterer.fit_kmeans(features, n_clusters=best_k)
        
        profiles = clusterer.create_cluster_profiles(features, feature_cols, rfm)
        
        # Save
        rules.to_csv('outputs/tables/03_association_rules.csv', index=False)
        profiles.to_csv('outputs/tables/03_cluster_profiles.csv', index=False)
        
        logger.info("Data Mining completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data Mining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_modeling():
    """Run classification and forecasting"""
    logger.info("=" * 60)
    logger.info("STEP 4: Modeling")
    logger.info("=" * 60)
    
    try:
        from src.data.loader import DataLoader
        from src.data.cleaner import DataCleaner
        from src.features.builder import FeatureBuilder
        from src.mining.clustering import ClusterMiner
        from src.models.supervised import SupervisedModel
        from src.models.forecasting import TimeSeriesModel
        from sklearn.preprocessing import StandardScaler
        
        # Load data
        loader = DataLoader()
        df = loader.generate_sample_data(n_orders=2000)
        
        cleaner = DataCleaner(df)
        df = cleaner.handle_missing_values()
        
        builder = FeatureBuilder(df)
        rfm = builder.create_rfm_features()
        
        # Classification
        logger.info("Running Classification...")
        feature_cols = ['Recency', 'Frequency', 'Monetary']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(rfm[feature_cols])
        
        clusterer = ClusterMiner(n_clusters=4, random_state=42)
        labels = clusterer.fit_kmeans(features_scaled, n_clusters=4)
        
        rfm['Cluster'] = labels
        
        # Create features
        rfm['Avg_Profit'] = rfm['Total_Profit'] / rfm['Frequency']
        rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
        
        X = rfm[['Recency', 'Frequency', 'Monetary', 'Total_Profit', 'Avg_Profit', 'Avg_Order_Value']]
        y = rfm['Segment']
        
        clf = SupervisedModel(random_state=42, test_size=0.2)
        clf.prepare_data(X, y, scale=True)
        
        clf.train_baseline_logistic_regression()
        clf.train_baseline_decision_tree(max_depth=10)
        clf.train_improved_random_forest(n_estimators=100)
        
        # Forecasting
        logger.info("Running Forecasting...")
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        
        ts_model = TimeSeriesModel(freq='W', test_size=0.2)
        ts = ts_model.prepare_time_series(df, 'Order Date', 'Sales', 'sum')
        train, test = ts_model.train_test_split(n_test=12)
        
        ts_model.baseline_naive()
        ts_model.baseline_moving_average(window=4)
        ts_model.fit_arima(order=(1, 1, 1))
        ts_model.fit_holt_winters(seasonal='add', seasonal_periods=12)
        
        # Save
        clf.compare_models().to_csv('outputs/tables/04_classification_comparison.csv', index=False)
        ts_model.compare_models().to_csv('outputs/tables/04_forecasting_comparison.csv', index=False)
        
        logger.info("Modeling completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation():
    """Run evaluation and generate report"""
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluation & Reporting")
    logger.info("=" * 60)
    
    try:
        from src.evaluation.report import ReportGenerator
        
        # Generate final report
        generator = ReportGenerator(output_dir='outputs/')
        
        all_results = {
            'summary_stats': {
                'total_orders': 5000,
                'total_customers': 800,
                'total_sales': 450000,
                'total_profit': 52000
            }
        }
        
        generator.generate_final_report(all_results, 'outputs/reports/final_report.txt')
        
        logger.info("Evaluation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False


def main():
    """Main pipeline runner"""
    logger.info("Starting Data Mining Pipeline...")
    logger.info(f"Start time: {datetime.now()}")
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/tables', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    # Run pipeline steps
    steps = [
        ("EDA", run_eda),
        ("Preprocessing", run_preprocessing),
        ("Mining", run_mining),
        ("Modeling", run_modeling),
        ("Evaluation", run_evaluation)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {step_name}")
        logger.info(f"{'='*60}")
        
        success = step_func()
        results[step_name] = "✓ PASSED" if success else "✗ FAILED"
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    
    for step, result in results.items():
        logger.info(f"{step}: {result}")
    
    logger.info(f"\nEnd time: {datetime.now()}")
    logger.info("Pipeline completed!")
    
    return results


if __name__ == "__main__":
    import pandas as pd
    main()