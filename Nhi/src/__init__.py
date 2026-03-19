"""
Superstore Sales Data Mining Project
Phân tích doanh số siêu thị
"""

__version__ = "1.0"
__author__ = "Data Mining Team"

# Import key modules for easy access
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import AssociationMiner
from src.mining.clustering import ClusterMiner
from src.models.supervised import SupervisedModel
from src.models.forecasting import TimeSeriesModel
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.report import ReportGenerator

__all__ = [
    "DataLoader",
    "DataCleaner",
    "FeatureBuilder",
    "AssociationMiner",
    "ClusterMiner",
    "SupervisedModel",
    "TimeSeriesModel",
    "ModelEvaluator",
    "ReportGenerator",
]