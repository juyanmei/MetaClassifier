"""
LASSO classifier implementation for metaClassifier.

This module contains the LASSO classifier with integrated feature selection.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from .base_model import BaseModel


class LassoClassifier(BaseModel):
    """LASSO classifier using LogisticRegression with L1 penalty."""
    
    # 明确指定分类器类型，满足新版本scikit-learn的要求
    _estimator_type = "classifier"
    
    def __init__(self,
                 random_state: int = 42,
                 max_iter: int = 5000,
                 n_jobs: int = 1,
                 tol: float = 1e-6,
                 C: float = 1.0,
                 warm_start: bool = False,
                 **kwargs):
        super().__init__("Lasso", **kwargs)
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.tol = tol
        self.C = C
        self.warm_start = warm_start

        self.clf_ = None
        self.feature_importance_ = None
        
    def set_params(self, **params):
        """Set parameters for the classifier."""
        # 处理所有可能的参数
        valid_params = ['C', 'max_iter', 'tol', 'n_jobs', 'random_state', 'warm_start']
        for param, value in params.items():
            if param in valid_params and hasattr(self, param):
                setattr(self, param, value)
        # 调用父类的 set_params 方法
        super().set_params(**params)
        return self
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = {
            'C': self.C,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'warm_start': self.warm_start
        }
        if deep:
            return {**params, **super().get_params(deep=deep)}
        return params
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> 'LassoClassifier':
        """Fit the LASSO classifier to the training data."""
        super().fit(X, y, **kwargs)
        
        # LogisticRegression with L1 penalty (Lasso for classification)
        self.clf_ = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=self.C,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            tol=self.tol,
            warm_start=self.warm_start
        )

        self.clf_.fit(X, y)

        # Feature importance as absolute coefficients
        try:
            self.feature_importance_ = np.abs(self.clf_.coef_).ravel()
        except Exception:
            self.feature_importance_ = None
        
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.clf_.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.clf_.predict_proba(X)
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self.feature_importance_
        
    def get_selected_features(self) -> List[str]:
        """Get names of selected features (non-zero coefficients)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting selected features")
            
        try:
            # 使用原始系数而不是绝对值来判断特征是否被选择
            coef = self.clf_.coef_.ravel()
            selected_mask = coef != 0
            return [self.feature_names_[i] for i in range(len(self.feature_names_)) if selected_mask[i]]
        except Exception as e:
            self.logger.warning(f"获取选中特征时出错: {e}")
            return []
        
    def get_coefficients(self) -> np.ndarray:
        """Get LASSO coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        try:
            return self.clf_.coef_.ravel()
        except Exception:
            return np.array([])
        
    def get_alpha(self) -> float:
        """Get the C value (inverse of regularization strength)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting C")
        try:
            return float(self.clf_.C)
        except Exception:
            return np.nan
