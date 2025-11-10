"""
Neural Network classifier implementation for metaClassifier.

This module contains the Neural Network classifier implementation.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

from .base_model import BaseModel


class NeuralNetworkClassifier(BaseModel):
    """Neural Network classifier implementation."""
    
    def __init__(self,
                 hidden_layer_sizes: tuple = (100,),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 learning_rate: str = 'constant',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 200,
                 random_state: int = 42,
                 **kwargs):
        super().__init__("NeuralNetwork", **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.nn_ = None
        self.feature_importance_ = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> 'NeuralNetworkClassifier':
        """Fit the Neural Network classifier to the training data."""
        super().fit(X, y, **kwargs)
        
        # Initialize Neural Network
        self.nn_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        # Fit the model
        self.nn_.fit(X, y)
        
        # Calculate feature importance (using input layer weights)
        self.feature_importance_ = self._calculate_feature_importance(X)
        
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.nn_.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.nn_.predict_proba(X)
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self.feature_importance_
        
    def _calculate_feature_importance(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate feature importance using input layer weights."""
        if hasattr(self.nn_, 'coefs_') and len(self.nn_.coefs_) > 0:
            # Use the absolute sum of weights from input layer
            input_weights = self.nn_.coefs_[0]  # Shape: (n_features, n_hidden_units)
            importance = np.sum(np.abs(input_weights), axis=1)
            return importance
        else:
            # Fallback: return uniform importance
            return np.ones(X.shape[1])
