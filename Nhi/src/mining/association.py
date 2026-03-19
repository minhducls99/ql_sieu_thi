"""
Module khai phá luật kết hợp (Association Rule Mining)
=====================================================
Sử dụng thuật toán Apriori để tìm các luật kết hợp trong dữ liệu giỏ hàng

Các khái niệm chính:
- Support: Tỷ lệ giao dịch chứa itemset / tổng giao dịch
- Confidence: Xác suất mua B khi đã mua A
- Lift: Hệ số tăng cường - cho biết A và B kết hợp tốt hơn ngẫu nhiên bao nhiêu
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssociationMiner:
    """
    Lớp khai phá luật kết hợp sử dụng thuật toán Apriori
    Dùng để tìm các mối quan hệ giữa các sản phẩm trong giỏ hàng
    """
    
    def __init__(self, min_support: float = 0.01, 
                 min_confidence: float = 0.3,
                 min_lift: float = 1.0):
        """
        Khởi tạo AssociationMiner
        
        Args:
            min_support: Ngưỡng hỗ trợ tối thiểu (mặc định 1% - itemset xuất hiện trong 1% giao dịch)
            min_confidence: Ngưỡng tin cậy tối thiểu (mặc định 30%)
            min_lift: Ngưỡng lift tối thiểu (mặc định 1.0 - không cải thiện)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.frequent_itemsets = None  # Các tập phổ biến tìm được
        self.rules = None              # Các luật kết hợp tìm được
    
    def fit(self, transactions: List[List[str]]) -> Dict[str, Any]:
        """
        Huấn luyện mô hình khai phá luật kết hợp trên dữ liệu giao dịch
        
        Args:
            transactions: Danh sách các giao dịch (mỗi giao dịch là danh sách các sản phẩm)
            
        Returns:
            Dictionary chứa frequent itemsets và các luật
        """
        logger.info(f"Fitting Apriori with min_support={self.min_support}, min_confidence={self.min_confidence}")
        
        # Chuyển đổi transactions sang ma trận nhị phân (one-hot encoding)
        # TransactionEncoder mã hóa mỗi giao dịch thành vector nhị phân
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Tìm các tập phổ biến (frequent itemsets) sử dụng thuật toán Apriori
        # Apriori hoạt động bằng cách tìm các itemset có support >= min_support
        self.frequent_itemsets = apriori(
            df, 
            min_support=self.min_support,
            use_colnames=True
        )
        
        if len(self.frequent_itemsets) == 0:
            logger.warning("No frequent itemsets found with given thresholds")
            return {'itemsets': pd.DataFrame(), 'rules': pd.DataFrame()}
        
        # Sinh các luật kết hợp từ các frequent itemsets
        # Luật có dạng: A -> B (nếu mua A thì sẽ mua B)
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence
        )
        
        # Lọc theo lift - chỉ giữ các luật có lift >= min_lift
        # Lift > 1 có nghĩa là A và B có mối quan hệ tích cực
        if len(self.rules) > 0:
            self.rules = self.rules[self.rules['lift'] >= self.min_lift]
        
        logger.info(f"Found {len(self.frequent_itemsets)} frequent itemsets and {len(self.rules)} rules")
        
        return {
            'itemsets': self.frequent_itemsets,
            'rules': self.rules
        }
    
    def get_top_rules(self, n: int = 20, sort_by: str = 'lift') -> pd.DataFrame:
        """
        Get top N association rules.
        
        Args:
            n: Number of top rules to return
            sort_by: Metric to sort by (support, confidence, lift)
            
        Returns:
            DataFrame with top rules
        """
        if self.rules is None or len(self.rules) == 0:
            return pd.DataFrame()
        
        top_rules = self.rules.sort_values(sort_by, ascending=False).head(n)
        
        # Simplify for readability
        result = pd.DataFrame({
            'Antecedent': top_rules['antecedents'].apply(lambda x: ', '.join(list(x))),
            'Consequent': top_rules['consequents'].apply(lambda x: ', '.join(list(x))),
            'Support': top_rules['support'].round(4),
            'Confidence': top_rules['confidence'].round(4),
            'Lift': top_rules['lift'].round(4)
        })
        
        return result
    
    def get_cross_sell_recommendations(self, item: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get cross-sell recommendations for a specific item.
        
        Args:
            item: Item to get recommendations for
            top_n: Number of recommendations
            
        Returns:
            List of (recommended_item, confidence) tuples
        """
        if self.rules is None or len(self.rules) == 0:
            return []
        
        # Find rules where the item is in antecedent
        rules_with_item = self.rules[
            self.rules['antecedents'].apply(lambda x: item in x)
        ]
        
        if len(rules_with_item) == 0:
            return []
        
        # Get top items by confidence
        recommendations = rules_with_item.nlargest(top_n, 'confidence')
        
        result = []
        for _, row in recommendations.iterrows():
            consequent_items = list(row['consequents'])
            for item in consequent_items:
                result.append((item, row['confidence']))
        
        return result[:top_n]
    
    def get_combo_recommendations(self, items: List[str]) -> List[Tuple[str, float]]:
        """
        Get recommendations for a combo of items.
        
        Args:
            items: List of items in the combo
            
        Returns:
            List of (recommended_item, confidence) tuples
        """
        if self.rules is None or len(self.rules) == 0:
            return []
        
        # Find rules where all combo items are in antecedent
        rules_with_combo = self.rules[
            self.rules['antecedents'].apply(lambda x: all(item in x for item in items))
        ]
        
        if len(rules_with_combo) == 0:
            return []
        
        # Get top by confidence
        recommendations = rules_with_combo.nlargest(5, 'confidence')
        
        result = []
        for _, row in recommendations.iterrows():
            consequent_items = list(row['consequents'])
            result.append((', '.join(consequent_items), row['confidence']))
        
        return result
    
    def get_rule_metrics(self) -> Dict[str, Any]:
        """
        Get summary metrics of the rules.
        
        Returns:
            Dictionary with rule metrics
        """
        if self.rules is None or len(self.rules) == 0:
            return {}
        
        return {
            'total_rules': len(self.rules),
            'avg_support': self.rules['support'].mean(),
            'avg_confidence': self.rules['confidence'].mean(),
            'avg_lift': self.rules['lift'].mean(),
            'max_lift': self.rules['lift'].max(),
            'unique_antecedents': len(self.rules['antecedents'].apply(lambda x: tuple(x)).unique()),
            'unique_consequents': len(self.rules['consequents'].apply(lambda x: tuple(x)).unique())
        }
    
    def generate_insights(self) -> List[str]:
        """
        Generate business insights from the rules.
        
        Returns:
            List of insight strings
        """
        insights = []
        
        if self.rules is None or len(self.rules) == 0:
            return ["No significant association rules found."]
        
        # Top rules by lift
        top_lift = self.rules.nlargest(5, 'lift')
        for _, row in top_lift.iterrows():
            ant = ', '.join(list(row['antecedents']))
            cons = ', '.join(list(row['consequents']))
            insights.append(
                f"Strong association: {ant} → {cons} (lift: {row['lift']:.2f}, confidence: {row['confidence']:.2%})"
            )
        
        # High confidence rules
        high_conf = self.rules[self.rules['confidence'] >= 0.7].nlargest(3, 'confidence')
        for _, row in high_conf.iterrows():
            ant = ', '.join(list(row['antecedents']))
            cons = ', '.join(list(row['consequents']))
            insights.append(
                f"High confidence combo: {ant} → {cons} (confidence: {row['confidence']:.2%})"
            )
        
        return insights


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
    basket = builder.create_basket_data(min_items=2)
    
    transactions = basket['Items'].tolist()
    
    miner = AssociationMiner(min_support=0.02, min_confidence=0.3, min_lift=1.0)
    result = miner.fit(transactions)
    
    print("\nTop 10 rules by lift:")
    print(miner.get_top_rules(10))
    
    print("\nRule metrics:")
    print(miner.get_rule_metrics())
    
    print("\nInsights:")
    for insight in miner.generate_insights():
        print(f"  - {insight}")