"""
Feature selection strategies for metaClassifier.

This module contains various feature selection methods and strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, 
    SelectFromModel, 
    RFECV,
    f_classif,
    mutual_info_classif
)

from .base import BaseFeatureSelector


class FeatureSelector(BaseFeatureSelector):
    """Main feature selector class."""
    
    def __init__(self, method: str = "selectkbest", **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.selector = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FeatureSelector':
        """Fit the feature selector."""
        if self.method == "selectkbest":
            self.selector = SelectKBestSelector(**self.config)
        elif self.method == "selectfrommodel":
            self.selector = SelectFromModelSelector(**self.config)
        elif self.method == "rfecv":
            self.selector = RFECVSelector(**self.config)
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
        
        self.selector.fit(X, y)
        self.selected_features_ = self.selector.get_support()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        if self.selector is None:
            raise ValueError("Selector must be fitted before transforming")
        return self.selector.transform(X)
        
    def get_support(self) -> np.ndarray:
        """Get a boolean mask of selected features."""
        return self.selected_features_


class FeatureSelectorFactory:
    """Factory for creating feature selectors."""
    
    @staticmethod
    def create_selector(method: str, **kwargs) -> BaseFeatureSelector:
        """Create a feature selector based on method name."""
        if method == "selectkbest":
            return SelectKBestSelector(**kwargs)
        elif method == "selectfrommodel":
            return SelectFromModelSelector(**kwargs)
        elif method == "rfecv":
            return RFECVSelector(**kwargs)
        elif method == "adaptive":
            return AdaptiveFeatureSelector(**kwargs)
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")


class SelectKBestSelector(BaseFeatureSelector):
    """SelectKBest feature selector."""
    
    def __init__(self, k: int = 10, score_func: str = "f_classif", **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.score_func = score_func
        self.selector = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'SelectKBestSelector':
        """Fit the selector to the data."""
        if self.score_func == "f_classif":
            score_func = f_classif
        elif self.score_func == "mutual_info":
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unsupported score function: {self.score_func}")
            
        self.selector = SelectKBest(score_func=score_func, k=self.k)
        self.selector.fit(X, y)
        self.selected_features_ = self.selector.get_support()
        self.feature_scores_ = self.selector.scores_
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from the data."""
        if self.selector is None:
            raise ValueError("Selector must be fitted before transforming")
        return pd.DataFrame(
            self.selector.transform(X),
            columns=X.columns[self.selected_features_],
            index=X.index
        )
        
    def get_support(self) -> np.ndarray:
        """Get a boolean mask of selected features."""
        return self.selected_features_


class SelectFromModelSelector(BaseFeatureSelector):
    """SelectFromModel feature selector."""
    
    def __init__(self, estimator, threshold: str = "median", **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.threshold = threshold
        self.selector = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'SelectFromModelSelector':
        """Fit the selector to the data."""
        self.selector = SelectFromModel(
            estimator=self.estimator,
            threshold=self.threshold
        )
        self.selector.fit(X, y)
        self.selected_features_ = self.selector.get_support()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from the data."""
        if self.selector is None:
            raise ValueError("Selector must be fitted before transforming")
        return pd.DataFrame(
            self.selector.transform(X),
            columns=X.columns[self.selected_features_],
            index=X.index
        )
        
    def get_support(self) -> np.ndarray:
        """Get a boolean mask of selected features."""
        return self.selected_features_


class RFECVSelector(BaseFeatureSelector):
    """RFECV feature selector."""
    
    def __init__(self, estimator, cv: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.cv = cv
        self.selector = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'RFECVSelector':
        """Fit the selector to the data."""
        self.selector = RFECV(
            estimator=self.estimator,
            cv=self.cv,
            scoring='roc_auc'
        )
        self.selector.fit(X, y)
        self.selected_features_ = self.selector.get_support()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from the data."""
        if self.selector is None:
            raise ValueError("Selector must be fitted before transforming")
        return pd.DataFrame(
            self.selector.transform(X),
            columns=X.columns[self.selected_features_],
            index=X.index
        )
        
    def get_support(self) -> np.ndarray:
        """Get a boolean mask of selected features."""
        return self.selected_features_


class AdaptiveFeatureSelector(BaseFeatureSelector):
    """Adaptive feature selector that combines multiple strategies."""
    
    def __init__(self, strategies: List[str], **kwargs):
        super().__init__(**kwargs)
        self.strategies = strategies
        self.selectors = {}
        self.selected_features_ = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'AdaptiveFeatureSelector':
        """Fit the adaptive selector to the data."""
        # Implementation will be added
        pass
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from the data."""
        # Implementation will be added
        pass
        
    def get_support(self) -> np.ndarray:
        """Get a boolean mask of selected features."""
        return self.selected_features_
