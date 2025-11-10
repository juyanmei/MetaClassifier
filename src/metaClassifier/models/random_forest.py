"""
Random Forest classifier implementation for metaClassifier.

This module contains the Random Forest classifier implementation.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

from .base_model import BaseModel


class RandomForestClassifier(BaseModel):
    """Random Forest classifier implementation."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float, None] = 'sqrt',
                 random_state: int = 42,
                 n_jobs: int = 1,
                 **kwargs):
        super().__init__("RandomForest", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.rf_ = None
        self.feature_importance_ = None
        
    def set_params(self, **params):
        """Set parameters for the classifier."""
        # 处理所有可能的参数
        valid_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                       'max_features', 'random_state', 'n_jobs']
        for param, value in params.items():
            if param in valid_params and hasattr(self, param):
                setattr(self, param, value)
        # 调用父类的 set_params 方法
        super().set_params(**params)
        return self
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        if deep:
            return {**params, **super().get_params(deep=deep)}
        return params
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> 'RandomForestClassifier':
        """Fit the Random Forest classifier to the training data."""
        super().fit(X, y, **kwargs)
        
        # Initialize Random Forest
        self.rf_ = SklearnRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Fit the model
        self.rf_.fit(X, y)
        
        # Store feature importance
        self.feature_importance_ = self.rf_.feature_importances_
        
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.rf_.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.rf_.predict_proba(X)
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self.feature_importance_
        
    def get_feature_names_by_importance(self, top_k: Optional[int] = None) -> List[str]:
        """Get feature names sorted by importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature names")
            
        # Get indices sorted by importance (descending)
        importance_indices = np.argsort(self.feature_importance_)[::-1]
        
        if top_k is not None:
            importance_indices = importance_indices[:top_k]
            
        return [self.feature_names_[i] for i in importance_indices]
