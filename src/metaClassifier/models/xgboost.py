"""
XGBoost分类器实现。

XGBoost是一个高效的梯度提升算法，适合高维数据。
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels

try:
    from xgboost import XGBClassifier as _XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    _XGBClassifier = None

from .base_model import BaseModel


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost分类器包装器。
    
    提供与sklearn兼容的接口。
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, subsample: float = 1.0,
                 colsample_bytree: float = 1.0, reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0, random_state: Optional[int] = None,
                 **kwargs):
        """
        初始化XGBoost分类器。
        
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            subsample: 样本采样比例
            colsample_bytree: 特征采样比例
            reg_alpha: L1正则化参数
            reg_lambda: L2正则化参数
            random_state: 随机种子
            **kwargs: 其他XGBoost参数
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
            
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        
        # 存储其他参数
        self.kwargs = kwargs
        
        self.xgb_classifier_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'XGBoostClassifier':
        """
        训练XGBoost分类器。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            
        Returns:
            self
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # 获取类别信息
        self.classes_ = unique_labels(y)
        
        # 创建XGBoost分类器
        self.xgb_classifier_ = _XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            **self.kwargs
        )
        
        # 训练
        self.xgb_classifier_.fit(X, y)
        
        # 保存特征信息
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
            
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测分类结果。
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测标签
        """
        if self.xgb_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        return self.xgb_classifier_.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测概率。
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率
        """
        if self.xgb_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        return self.xgb_classifier_.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性。
        
        Returns:
            特征重要性数组
        """
        if self.xgb_classifier_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
            
        return self.xgb_classifier_.feature_importances_
    
    def get_support(self, threshold: float = 0.0) -> np.ndarray:
        """
        获取特征选择支持。
        
        Args:
            threshold: 重要性阈值
            
        Returns:
            布尔数组，True表示特征被选中
        """
        if self.xgb_classifier_ is None:
            raise ValueError("Model must be fitted before getting support")
            
        importance = self.get_feature_importance()
        return importance > threshold
    
    def get_params(self, deep: bool = True) -> dict:
        """获取参数。"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params) -> 'XGBoostClassifier':
        """设置参数。"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self
