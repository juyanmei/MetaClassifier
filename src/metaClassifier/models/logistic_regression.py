"""
逻辑回归分类器实现（sklearn 包装）。
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels

from .base_model import BaseModel


class LogisticRegressionClassifier(BaseModel):
    """
    对 sklearn LogisticRegression 的轻量封装，提供统一接口：
    - predict
    - predict_proba
    - get_feature_importance（|coef|）
    """

    def __init__(
        self,
        penalty: str = 'l2',
        C: float = 1.0,
        solver: str = 'liblinear',
        max_iter: int = 1000,
        tol: float = 1e-4,
        class_weight: Optional[Union[str, dict]] = None,
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__("LogisticRegression", **kwargs)
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state

        self.model_: Optional[LogisticRegression] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegressionClassifier only supports binary classification")

        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)

        # 调用BaseModel的fit方法
        super().fit(X, y)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = check_array(X, accept_sparse=True)
        return self.model_.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = check_array(X, accept_sparse=True)
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
        coef = getattr(self.model_, 'coef_', None)
        if coef is None:
            return np.zeros(self.n_features_in_ or 0)
        # coef_ shape: (1, n_features) for binary
        return np.abs(coef.ravel())

    def get_params(self, deep: bool = True) -> dict:
        return {
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


