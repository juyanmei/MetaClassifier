"""
Feature engineering utilities for metaClassifier.

This module contains various feature engineering techniques for microbiome data.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from ..core.base import BasePreprocessor
from ..utils.logger import get_logger


class FeatureEngineer(BasePreprocessor):
    """Feature engineering utilities for microbiome data."""
    
    def __init__(self, 
                 create_ratios: bool = True,
                 create_interactions: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.create_ratios = create_ratios
        self.create_interactions = create_interactions
        self.logger = get_logger("FeatureEngineer")
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'FeatureEngineer':
        """Fit the feature engineer to the data."""
        self.logger.info("Fitting feature engineer...")
        
        # Store original feature names
        self.feature_names_ = X.columns.tolist()
        
        self.is_fitted = True
        self.logger.info("Feature engineer fitted successfully")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by creating engineered features."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transforming")
            
        self.logger.info("Creating engineered features...")
        
        X_engineered = X.copy()
        
        if self.create_ratios:
            X_engineered = self._create_ratio_features(X_engineered)
            
        if self.create_interactions:
            X_engineered = self._create_interaction_features(X_engineered)
            
        self.logger.info(f"Created {X_engineered.shape[1] - X.shape[1]} engineered features")
        return X_engineered
        
    def _create_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features between top features."""
        # Get top features by variance
        top_features = X.var().nlargest(10).index.tolist()
        
        ratio_features = []
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                # Create ratio feature
                ratio_name = f"{feat1}_div_{feat2}"
                ratio_values = X[feat1] / (X[feat2] + 1e-8)  # Add small value to avoid division by zero
                ratio_features.append((ratio_name, ratio_values))
        
        # Add ratio features to DataFrame
        for name, values in ratio_features:
            X[name] = values
            
        return X
        
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between top features."""
        # Get top features by variance
        top_features = X.var().nlargest(5).index.tolist()
        
        interaction_features = []
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                # Create interaction feature
                interaction_name = f"{feat1}_x_{feat2}"
                interaction_values = X[feat1] * X[feat2]
                interaction_features.append((interaction_name, interaction_values))
        
        # Add interaction features to DataFrame
        for name, values in interaction_features:
            X[name] = values
            
        return X
