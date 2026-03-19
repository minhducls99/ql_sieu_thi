"""
Module phân cụm (Clustering)
============================
Các thuật toán phân cụm được sử dụng:
- K-Means: Phân cụm theo khoảng cách trung tâm
- HAC (Hierarchical Agglomerative Clustering): Phân cụm phân cấp
- DBSCAN: Phân cụm dựa trên mật độ

Các chỉ số đánh giá:
- Silhouette Score: Đánh giá độ tách biệt giữa các cụm (-1 đến 1, càng cao càng tốt)
- Davies-Bouldin Index (DBI): Đánh giá độ tương tự trong cụm (càng thấp càng tốt)
- Calinski-Harabasz Index: Tỷ lệ giữa phương sai giữa cụm và trong cụm (càng cao càng tốt)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterMiner:
    """
    Lớp phân cụm khách hàng sử dụng các thuật toán K-Means, HAC, DBSCAN
    Dùng để phân khúc khách hàng thành các nhóm có đặc điểm tương tự
    """
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Khởi tạo ClusterMiner
        
        Args:
            n_clusters: Số cụm mong muốn
            random_state: Hạt giống ngẫu nhiên để tái tạo kết quả
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.scaler = StandardScaler()  # Chuẩn hóa dữ liệu trước khi phân cụm
        self.cluster_profiles = None
    
    def fit_kmeans(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Huấn luyện K-Means clustering
        
        K-Means hoạt động:
        1. Khởi tạo k centroid ngẫu nhiên
        2. Gán mỗi điểm cho cụm gần nhất
        3. Cập nhật centroid = trung bình các điểm trong cụm
        4. Lặp lại bước 2-3 cho đến khi hội tụ
        
        Args:
            features: Ma trận đặc trưng
            n_clusters: Số cụm (ghi đè self.n_clusters)
            
        Returns:
            Nhãn cụm của mỗi điểm dữ liệu
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
            
        logger.info(f"Fitting K-Means with {n_clusters} clusters...")
        
        self.features = features  # Lưu features để đánh giá sau
        
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.labels = self.model.fit_predict(features)
        self.n_clusters = n_clusters
        
        logger.info(f"K-Means clustering complete. Silhouette: {self.get_silhouette_score():.4f}")
        
        return self.labels
    
    def fit_hac(self, features: np.ndarray, n_clusters: Optional[int] = None, 
                linkage: str = 'ward') -> np.ndarray:
        """
        Fit Hierarchical Agglomerative Clustering.
        
        Args:
            features: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage method (ward, complete, average)
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
            
        logger.info(f"Fitting HAC with {n_clusters} clusters using {linkage} linkage...")
        
        self.features = features  # Store features for evaluation
        
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        self.labels = self.model.fit_predict(features)
        
        logger.info(f"HAC clustering complete. Silhouette: {self.get_silhouette_score():.4f}")
        
        return self.labels
    
    def fit_dbscan(self, features: np.ndarray, eps: float = 0.5, 
                   min_samples: int = 5) -> np.ndarray:
        """
        Fit DBSCAN clustering.
        
        Args:
            features: Feature matrix
            eps: Maximum distance between points
            min_samples: Minimum points in a cluster
            
        Returns:
            Cluster labels
        """
        logger.info(f"Fitting DBSCAN with eps={eps}, min_samples={min_samples}...")
        
        self.features = features  # Store features for evaluation
        
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = self.model.fit_predict(features)
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        logger.info(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")
        
        if n_clusters > 1:
            logger.info(f"Silhouette (excluding noise): {self.get_silhouette_score(exclude_noise=True):.4f}")
        
        return self.labels
    
    def find_optimal_k(self, features: np.ndarray, 
                       k_range: range = range(2, 11),
                       method: str = 'kmeans') -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method and silhouette.
        
        Args:
            features: Feature matrix
            k_range: Range of K values to try
            method: Clustering method (kmeans, hac)
            
        Returns:
            Dictionary with K values and metrics
        """
        logger.info(f"Finding optimal K in range {list(k_range)}...")
        
        results = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in k_range:
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=k)
            
            labels = model.fit_predict(features)
            
            # Calculate metrics
            if len(set(labels)) > 1:
                sil = silhouette_score(features, labels)
                dbi = davies_bouldin_score(features, labels)
                ch = calinski_harabasz_score(features, labels)
            else:
                sil, dbi, ch = 0, float('inf'), 0
            
            results['k'].append(k)
            results['silhouette'].append(sil)
            results['davies_bouldin'].append(dbi)
            results['calinski_harabasz'].append(ch)
            
            if method == 'kmeans':
                if hasattr(model, 'inertia_'):
                    results['inertia'].append(model.inertia_)
                else:
                    results['inertia'].append(0)
            else:
                results['inertia'].append(0)
            
            logger.info(f"K={k}: Silhouette={sil:.4f}, DBI={dbi:.4f}, CH={ch:.4f}")
        
        # Find best K by silhouette
        best_k = results['k'][np.argmax(results['silhouette'])]
        logger.info(f"Best K by silhouette: {best_k}")
        
        return results
    
    def get_silhouette_score(self, exclude_noise: bool = False) -> float:
        """
        Calculate silhouette score.
        
        Args:
            exclude_noise: Whether to exclude noise points (for DBSCAN)
            
        Returns:
            Silhouette score
        """
        if self.labels is None:
            return 0
        
        labels = self.labels
        if exclude_noise and -1 in labels:
            mask = labels != -1
            if len(set(labels[mask])) > 1:
                return silhouette_score(self.features[mask], labels[mask])
            return 0
        
        if len(set(labels)) > 1:
            return silhouette_score(self.features, labels)
        return 0
    
    def create_cluster_profiles(self, features: np.ndarray, 
                                feature_names: List[str],
                                original_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create detailed profiles for each cluster.
        
        Args:
            features: Scaled feature matrix
            feature_names: Names of features
            original_data: Original DataFrame to get cluster statistics
            
        Returns:
            DataFrame with cluster profiles
        """
        if self.labels is None:
            raise ValueError("Clustering not performed. Call fit first.")
        
        self.features = features
        
        # Create profile DataFrame
        profiles = pd.DataFrame(features, columns=feature_names)
        profiles['Cluster'] = self.labels
        
        # Calculate cluster statistics
        cluster_stats = profiles.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
        
        # Simpler summary
        summary = profiles.groupby('Cluster').agg({
            name: 'mean' for name in feature_names
        }).reset_index()
        
        # Add cluster sizes
        cluster_sizes = pd.Series(self.labels).value_counts().sort_index()
        summary['Size'] = cluster_sizes.values
        summary['Percentage'] = (summary['Size'] / len(self.labels) * 100).round(2)
        
        self.cluster_profiles = summary
        return summary
    
    def get_cluster_descriptions(self) -> Dict[int, str]:
        """
        Generate human-readable cluster descriptions.
        
        Returns:
            Dictionary mapping cluster IDs to descriptions
        """
        if self.cluster_profiles is None:
            raise ValueError("No cluster profiles. Call create_cluster_profiles first.")
        
        descriptions = {}
        
        for _, row in self.cluster_profiles.iterrows():
            cluster_id = int(row['Cluster'])
            size = row['Size']
            pct = row['Percentage']
            
            # Determine key characteristics
            chars = []
            
            # Check Recency (lower is better - more recent)
            if 'Recency' in row:
                if row['Recency'] < self.cluster_profiles['Recency'].median():
                    chars.append('Recent buyers')
                else:
                    chars.append('Inactive buyers')
            
            # Check Frequency
            if 'Frequency' in row:
                if row['Frequency'] > self.cluster_profiles['Frequency'].median():
                    chars.append('Frequent')
                else:
                    chars.append('Occasional')
            
            # Check Monetary
            if 'Monetary' in row:
                if row['Monetary'] > self.cluster_profiles['Monetary'].median():
                    chars.append('High-value')
                else:
                    chars.append('Low-value')
            
            desc = f"Cluster {cluster_id}: {', '.join(chars)} ({size} customers, {pct}%)"
            descriptions[cluster_id] = desc
        
        return descriptions
    
    def visualize_clusters(self, features: np.ndarray, 
                          method: str = 'pca',
                          save_path: Optional[str] = None):
        """
        Visualize clusters in 2D using PCA or direct visualization.
        
        Args:
            features: Feature matrix
            method: Visualization method (pca, tsne, direct)
            save_path: Path to save figure
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        coords = reducer.fit_transform(features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                            c=self.labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'Customer Clusters ({method.upper()})')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to {save_path}")
        
        plt.close()
    
    def visualize_elbow(self, results: Dict[str, List], 
                       save_path: Optional[str] = None):
        """
        Visualize elbow curve.
        
        Args:
            results: Results from find_optimal_k
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Inertia (only for K-Means)
        if results['inertia']:
            axes[0].plot(results['k'], results['inertia'], 'bo-')
            axes[0].set_xlabel('Number of Clusters (K)')
            axes[0].set_ylabel('Inertia')
            axes[0].set_title('Elbow Method')
            axes[0].grid(True, alpha=0.3)
        
        # Silhouette
        axes[1].plot(results['k'], results['silhouette'], 'go-')
        axes[1].set_xlabel('Number of Clusters (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        
        # Davies-Bouldin
        axes[2].plot(results['k'], results['davies_bouldin'], 'ro-')
        axes[2].set_xlabel('Number of Clusters (K)')
        axes[2].set_ylabel('Davies-Bouldin Index')
        axes[2].set_title('Davies-Bouldin Index (lower is better)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Elbow plot saved to {save_path}")
        
        plt.close()
    
    def evaluate_clustering(self, features: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality.
        
        Args:
            features: Feature matrix
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.labels is None:
            return {}
        
        n_clusters = len(set(self.labels))
        
        if n_clusters < 2:
            return {'error': 'Need at least 2 clusters'}
        
        metrics = {
            'n_clusters': n_clusters,
            'silhouette': silhouette_score(features, self.labels),
            'davies_bouldin': davies_bouldin_score(features, self.labels),
            'calinski_harabasz': calinski_harabasz_score(features, self.labels)
        }
        
        return metrics


# Demo usage
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.cleaner import DataCleaner
    from src.features.builder import FeatureBuilder
    
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
    
    # Prepare features
    feature_cols = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    features = scaler.fit_transform(rfm[feature_cols])
    
    # Find optimal K
    clusterer = ClusterMiner(n_clusters=4, random_state=42)
    results = clusterer.find_optimal_k(features, range(2, 8))
    
    print("\nOptimal K results:")
    print(pd.DataFrame(results))
    
    # Fit with optimal K
    clusterer.fit_kmeans(features, n_clusters=4)
    
    # Create profiles
    profiles = clusterer.create_cluster_profiles(features, feature_cols, rfm)
    print("\nCluster profiles:")
    print(profiles)
    
    print("\nCluster descriptions:")
    for cid, desc in clusterer.get_cluster_descriptions().items():
        print(f"  {desc}")