"""
Base classes and interfaces for metaClassifier.

This module defines the fundamental interfaces that all components must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Enumeration of supported task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class CVStrategy(Enum):
    """Enumeration of supported cross-validation strategies."""
    REPEATED_KFOLD = "repeated_kfold"
    LOCO = "loco"  # Leave-One-Cohort-Out
    STRATIFIED_KFOLD = "stratified_kfold"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    task_type: TaskType
    hyperparameters: Dict[str, Any]
    feature_selection: Optional[Dict[str, Any]] = None
    preprocessing: Optional[Dict[str, Any]] = None


@dataclass
class CVConfig:
    """Configuration for cross-validation."""
    strategy: CVStrategy
    outer_folds: int = 5
    inner_folds: int = 3
    n_repeats: int = 1
    random_state: int = 42
    n_jobs: int = 1


@dataclass
class AdaptiveFilterConfig:
    """Configuration for adaptive variance filtering."""
    enabled: bool = True
    min_q: float = 0.5      # 最小过滤50%
    max_q: float = 0.95     # 最大过滤95%
    r_mid: float = 1.0      # p/n比率中点1.0
    steepness: float = 2.0  # S型曲线陡峭度2.0


@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment."""
    model: ModelConfig
    cv: CVConfig
    adaptive_filter: AdaptiveFilterConfig
    feature_selection: Optional['FeatureSelectionConfig'] = None
    data_paths: Dict[str, str] = None
    output_dir: str = "./results"
    verbose: bool = True
    search_method: str = 'grid'  # 超参数搜索方法


class BaseModel(ABC):
    """Base class for all models in metaClassifier."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        self.is_fitted = False
        self.feature_names_ = None
        self.classes_ = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Fit the model to the training data."""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        pass
        
    @abstractmethod
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        pass
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return self.config.copy() if deep else self.config
        
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        self.config.update(params)
        return self


class BaseEvaluator(ABC):
    """Base class for all evaluators in metaClassifier."""
    
    def __init__(self, config: CVConfig):
        self.config = config
        self.results_ = None
        
    @abstractmethod
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        models: Dict[str, BaseModel],
        cohort_info: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate models using cross-validation."""
        pass
        
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Get evaluation results."""
        pass


class BaseClassifier(ABC):
    """Base class for all classifiers in metaClassifier."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.models_ = {}
        self.evaluator_ = None
        self.results_ = None
        
    @abstractmethod
    def fit(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        cohort_info: Optional[np.ndarray] = None
    ) -> 'BaseClassifier':
        """Fit the classifier to the training data."""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        pass
        
    @abstractmethod
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        cohort_info: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate the classifier using cross-validation."""
        pass


class BasePreprocessor(ABC):
    """Base class for all preprocessors in metaClassifier."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'BasePreprocessor':
        """Fit the preprocessor to the data."""
        pass
        
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)


class BaseFeatureSelector(ABC):
    """Base class for all feature selectors in metaClassifier."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.selected_features_ = None
        self.feature_scores_ = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'BaseFeatureSelector':
        """Fit the feature selector to the data."""
        pass
        
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from the data."""
        pass
        
    @abstractmethod
    def get_support(self) -> np.ndarray:
        """Get a boolean mask of selected features."""
        pass
        
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)
