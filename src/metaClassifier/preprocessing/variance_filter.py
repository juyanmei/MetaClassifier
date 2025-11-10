"""
Adaptive variance filtering for metaClassifier.

This module contains the adaptive variance filtering implementation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..core.base import BasePreprocessor, AdaptiveFilterConfig
from ..utils.logger import get_logger


class AdaptiveVarianceFilter(BasePreprocessor):
    """Adaptive variance filter that adjusts filtering intensity based on p/n ratio."""
    
    def __init__(self, config: AdaptiveFilterConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.logger = get_logger("AdaptiveVarianceFilter")
        self.removed_features_ = None
        self.filter_info_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'AdaptiveVarianceFilter':
        """Fit the adaptive variance filter to the data."""
        if not self.config.enabled:
            self.logger.info("Adaptive variance filtering disabled")
            return self
            
        self.logger.info("Fitting adaptive variance filter...")
        
        # Calculate p/n ratio
        p_n_ratio = X.shape[1] / X.shape[0]
        
        # Calculate adaptive quantile
        adaptive_quantile = self._calculate_adaptive_quantile(
            n_features=X.shape[1],
            n_samples=X.shape[0]
        )
        
        # Calculate variance threshold
        feature_variances = X.var()
        remove_percentile = adaptive_quantile * 100  # 转换为百分位数
        threshold = np.percentile(feature_variances, remove_percentile)
        
        # Identify features to remove
        self.removed_features_ = feature_variances[feature_variances < threshold].index.tolist()
        
        # Store filter information
        self.filter_info_ = {
            'enabled': True,
            'original_features': X.shape[1],
            'removed_features': len(self.removed_features_),
            'p_n_ratio': p_n_ratio,
            'adaptive_quantile': adaptive_quantile,
            'threshold': threshold
        }
        
        self.is_fitted = True
        self.logger.info(f"Adaptive variance filter fitted: {self.filter_info_}")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by removing low variance features."""
        if not self.is_fitted:
            raise ValueError("Filter must be fitted before transforming")
            
        if not self.config.enabled or self.removed_features_ is None:
            return X.copy()
            
        # Remove low variance features
        X_filtered = X.drop(columns=self.removed_features_)
        
        self.logger.info(f"Removed {len(self.removed_features_)} low variance features")
        return X_filtered
        
    def filter_features(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Filter features and return filtered data with information."""
        X_filtered = self.fit_transform(X, y)
        return X_filtered, self.filter_info_ or {}
        
    def _calculate_adaptive_quantile(self, n_features: int, n_samples: int) -> float:
        """Calculate adaptive quantile based on p/n ratio."""
        r = n_features / n_samples
        
        min_q = self.config.min_q
        max_q = self.config.max_q
        r_mid = self.config.r_mid
        steepness = self.config.steepness
        
        # 强制使用严格的过滤参数（针对高维小样本问题优化）
        min_q = 0.5      # 最小过滤50%
        max_q = 0.95     # 最大过滤95%
        r_mid = 1.0      # p/n比率中点1.0
        steepness = 2.0  # S型曲线陡峭度2.0
        
        # Debug logging
        self.logger.info(f"使用严格过滤参数: min_q={min_q}, max_q={max_q}, r_mid={r_mid}, steepness={steepness}")
        
        # Use sigmoid function for smooth transition
        logistic_part = 1 / (1 + np.exp(-steepness * (r - r_mid)))
        
        # Map from [0, 1] to [min_q, max_q]
        adaptive_quantile = min_q + (max_q - min_q) * logistic_part
        
        self.logger.info(f"计算结果: r={r:.2f}, logistic_part={logistic_part:.6f}, adaptive_quantile={adaptive_quantile:.6f}")
        
        return adaptive_quantile
