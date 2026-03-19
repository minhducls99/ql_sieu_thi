"""
Module học có giám sát (Supervised Learning)
============================================
Các mô hình phân loại được sử dụng:
- Logistic Regression: Hồi quy logistic - mô hình phân loại nhị phân
- Decision Tree: Cây quyết định - phân loại dựa trên các quy tắc
- Random Forest: Rừng ngẫu nhiên - tập hợp nhiều cây quyết định
- Gradient Boosting: Tăng cường gradient - xây dựng mô hình theo từng bước

Các chỉ số đánh giá:
- Accuracy: Tỷ lệ dự đoán đúng
- Precision: Tỷ lệ dự đoán đúng trong các dự đoán dương
- Recall: Tỷ lệ thực tế dương được dự đoán đúng
- F1-Score: Trung bình điều hòa của Precision và Recall
- ROC-AUC: Diện tích dưới đường cong ROC
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, classification_report,
                            roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupervisedModel:
    """
    Lớp mô hình học có giám sát cho bài toán dự đoán phân khúc khách hàng
    Bao gồm: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
    """
    
    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        """
        Khởi tạo SupervisedModel
        
        Args:
            random_state: Hạt giống ngẫu nhiên để tái tạo kết quả
            test_size: Tỷ lệ dữ liệu dùng để test (mặc định 20%)
        """
        self.random_state = random_state
        self.test_size = test_size
        self.models = {}  # Lưu trữ các mô hình đã huấn luyện
        self.scaler = StandardScaler()  # Chuẩn hóa dữ liệu
        self.label_encoder = LabelEncoder()  # Mã hóa nhãn
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}  # Lưu kết quả đánh giá
    
    def prepare_data(self, features: pd.DataFrame, target: pd.Series,
                    scale: bool = True) -> Tuple:
        """
        Chuẩn bị dữ liệu cho mô hình
        
        Args:
            features: DataFrame chứa các đặc trưng
            target: Series chứa biến mục tiêu
            scale: Có chuẩn hóa dữ liệu không
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for modeling...")
        
        # Mã hóa biến mục tiêu (chuyển từ text sang số)
        y = self.label_encoder.fit_transform(target)
        
        # Chia dữ liệu thành train và test
        # stratify=y đảm bảo tỷ lệ các lớp trong train và test giống nhau
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        # Chuẩn hóa dữ liệu (quan trọng cho Logistic Regression)
        if scale:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_baseline_logistic_regression(self) -> Dict[str, Any]:
        """
        Train Logistic Regression baseline model.
        
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Logistic Regression...")
        
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Metrics
        metrics = self._calculate_metrics(y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                   cv=5, scoring='f1_macro')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        self.models['LogisticRegression'] = model
        self.results['LogisticRegression'] = metrics
        
        logger.info(f"Logistic Regression - F1 Macro: {metrics['f1_macro']:.4f}")
        
        return {'model': model, 'metrics': metrics}
    
    def train_baseline_decision_tree(self, max_depth: int = 10) -> Dict[str, Any]:
        """
        Train Decision Tree baseline model.
        
        Args:
            max_depth: Maximum depth of tree
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training Decision Tree (max_depth={max_depth})...")
        
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=self.random_state,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Metrics
        metrics = self._calculate_metrics(y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                   cv=5, scoring='f1_macro')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        self.models['DecisionTree'] = model
        self.results['DecisionTree'] = metrics
        
        logger.info(f"Decision Tree - F1 Macro: {metrics['f1_macro']:.4f}")
        
        return {'model': model, 'metrics': metrics}
    
    def train_improved_random_forest(self, n_estimators: int = 100,
                                    max_depth: int = 15) -> Dict[str, Any]:
        """
        Train Random Forest improved model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Metrics
        metrics = self._calculate_metrics(y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                   cv=5, scoring='f1_macro')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        self.models['RandomForest'] = model
        self.results['RandomForest'] = metrics
        
        logger.info(f"Random Forest - F1 Macro: {metrics['f1_macro']:.4f}")
        
        return {'model': model, 'metrics': metrics}
    
    def _calculate_metrics(self, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision_macro': precision_score(self.y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(self.y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(self.y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC (one-vs-rest)
        try:
            metrics['roc_auc'] = roc_auc_score(
                self.y_test, y_pred_proba, multi_class='ovr', average='macro'
            )
        except:
            metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def get_feature_importance(self, model_name: str = 'RandomForest') -> pd.DataFrame:
        """
        Get feature importance from a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def plot_confusion_matrix(self, model_name: str = 'RandomForest',
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            save_path: Path to save figure
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not trained")
        
        metrics = self.results[model_name]
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_roc_curve(self, model_name: str = 'RandomForest',
                     save_path: Optional[str] = None):
        """
        Plot ROC curve for a model.
        
        Args:
            model_name: Name of the model
            save_path: Path to save figure
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(self.X_test)
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, i], pos_label=i)
            auc = roc_auc_score(self.y_test == i, y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_feature_importance(self, model_name: str = 'RandomForest',
                              save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            model_name: Name of the model
            save_path: Path to save figure
        """
        importance_df = self.get_feature_importance(model_name)
        
        if len(importance_df) == 0:
            logger.warning(f"No feature importance for {model_name}")
            return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            row.update({k: v for k, v in metrics.items() if k != 'confusion_matrix'})
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('f1_macro', ascending=False)
        
        return df
    
    def analyze_errors(self, model_name: str = 'RandomForest') -> Dict[str, Any]:
        """
        Analyze classification errors.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with error analysis
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Find most confused pairs
        errors = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    errors.append({
                        'True': self.label_encoder.classes_[i],
                        'Predicted': self.label_encoder.classes_[j],
                        'Count': cm[i, j]
                    })
        
        errors_df = pd.DataFrame(errors).sort_values('Count', ascending=False)
        
        # Calculate per-class metrics
        per_class = classification_report(self.y_test, y_pred, 
                                         target_names=self.label_encoder.classes_,
                                         output_dict=True)
        
        return {
            'confusion_pairs': errors_df,
            'per_class_report': per_class,
            'total_errors': (y_pred != self.y_test).sum()
        }
    
    def save_model(self, model_name: str, path: str):
        """
        Save a trained model.
        
        Args:
            model_name: Name of the model
            path: Path to save model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        joblib.dump({
            'model': self.models[model_name],
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }, path)
        
        logger.info(f"Model {model_name} saved to {path}")
    
    def load_model(self, path: str) -> Any:
        """
        Load a trained model.
        
        Args:
            path: Path to load model from
            
        Returns:
            Loaded model
        """
        data = joblib.load(path)
        self.models['loaded'] = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.feature_names = data['feature_names']
        
        return data['model']


# Demo usage
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.cleaner import DataCleaner
    from src.features.builder import FeatureBuilder
    from src.mining.clustering import ClusterMiner
    from sklearn.preprocessing import StandardScaler
    
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
    
    builder = FeatureBuilder(cleaned)
    rfm = builder.create_rfm_features()
    
    # Cluster for labels
    feature_cols = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(rfm[feature_cols])
    
    clusterer = ClusterMiner(n_clusters=4, random_state=42)
    labels = clusterer.fit_kmeans(features_scaled, n_clusters=4)
    
    # Prepare for classification
    rfm['Cluster'] = labels
    
    # Create additional features
    rfm['Total_Profit'] = rfm['Total_Profit']
    rfm['Avg_Profit'] = rfm['Total_Profit'] / rfm['Frequency']
    rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
    
    X = rfm[['Recency', 'Frequency', 'Monetary', 'Total_Profit', 'Avg_Profit', 'Avg_Order_Value']]
    y = rfm['Segment']
    
    # Train models
    clf = SupervisedModel(random_state=42, test_size=0.2)
    clf.prepare_data(X, y)
    
    clf.train_baseline_logistic_regression()
    clf.train_baseline_decision_tree(max_depth=10)
    clf.train_improved_random_forest(n_estimators=100)
    
    print("\nModel Comparison:")
    print(clf.compare_models())
    
    print("\nFeature Importance (RandomForest):")
    print(clf.get_feature_importance('RandomForest'))