"""
Data preprocessing utilities for metaClassifier.

This module handles data preprocessing and transformation.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from ..core.base import BasePreprocessor
from ..utils.logger import get_logger


class DataPreprocessor(BasePreprocessor):
    """Minimal data preprocessor for microbiome data."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = get_logger("DataPreprocessor")
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'DataPreprocessor':
        """Fit the preprocessor to the data."""
        self.logger.info("Fitting minimal data preprocessor...")
        
        # 检查数据质量
        self._check_data_quality(X)
        
        self.is_fitted = True
        self.logger.info("Data preprocessor fitted successfully")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        self.logger.info("Transforming data...")
        
        # 直接返回数据，不进行任何变换
        # 常数特征移除和CLR转换已在DataLoader中完成
        X_processed = X.copy()
        
        self.logger.info("Data transformation completed")
        return X_processed
        
    def _check_data_quality(self, X: pd.DataFrame) -> None:
        """Check data quality and log warnings if needed."""
        # 检查缺失值
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            self.logger.warning(f"Found {missing_count} missing values in the data")
        
        # 检查无穷值
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self.logger.warning(f"Found {inf_count} infinite values in the data")
        
        # 检查数据范围
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            min_val = X[numeric_cols].min().min()
            max_val = X[numeric_cols].max().max()
            self.logger.info(f"Data range: [{min_val:.6f}, {max_val:.6f}]")
        
        self.logger.info(f"Data shape: {X.shape[0]} samples x {X.shape[1]} features")
