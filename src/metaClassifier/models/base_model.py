"""
Base model implementation for metaClassifier.

This module contains the base model class that all models inherit from.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..core.base import BaseModel as MetaClassifierBaseModel


class BaseModel(MetaClassifierBaseModel, ClassifierMixin, BaseEstimator):
    """Base model class for metaClassifier models."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, **kwargs) -> 'BaseModel':
        """Fit the model to the training data."""
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = None
        self.classes_ = np.unique(y)
        self.is_fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        # Implementation will be provided by subclasses
        pass
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        # Implementation will be provided by subclasses
        pass
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        # Implementation will be provided by subclasses
        return None
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return super().get_params(deep=deep)
