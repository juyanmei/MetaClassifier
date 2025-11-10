"""
GaussianNB分类器实现。

高斯朴素贝叶斯分类器，适合小样本数据。
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import GaussianNB as _GaussianNB

from .base_model import BaseModel


class GaussianNBClassifier(BaseEstimator, ClassifierMixin):
    """
    GaussianNB分类器包装器。
    
    提供与sklearn兼容的接口。
    """
    
    def __init__(self, var_smoothing: float = 1e-9, random_state: Optional[int] = None, **kwargs):
        """
        初始化GaussianNB分类器。
        
        Args:
            var_smoothing: 方差平滑参数
            random_state: 随机种子（GaussianNB不支持，但为了接口一致性保留）
            **kwargs: 其他GaussianNB参数
        """
        self.var_smoothing = var_smoothing
        self.random_state = random_state  # 虽然GaussianNB不支持，但保留接口一致性
        
        # 存储其他参数
        self.kwargs = kwargs
        
        self.gnb_classifier_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'GaussianNBClassifier':
        """
        训练GaussianNB分类器。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            
        Returns:
            self
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # 获取类别信息
        self.classes_ = unique_labels(y)
        
        # 创建GaussianNB分类器
        self.gnb_classifier_ = _GaussianNB(
            var_smoothing=self.var_smoothing,
            **self.kwargs
        )
        
        # 训练
        self.gnb_classifier_.fit(X, y)
        
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
        if self.gnb_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        return self.gnb_classifier_.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测概率。
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率
        """
        if self.gnb_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        return self.gnb_classifier_.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性。
        
        注意：GaussianNB没有直接的特征重要性，返回基于方差的估计。
        
        Returns:
            特征重要性数组
        """
        if self.gnb_classifier_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
            
        # 基于方差的特征重要性估计
        # 方差越小，特征越重要（更稳定）
        variances = self.gnb_classifier_.var_
        if len(variances.shape) == 2:
            # 多类别情况，取平均方差
            avg_variance = np.mean(variances, axis=0)
        else:
            avg_variance = variances
            
        # 转换为重要性（方差越小，重要性越高）
        importance = 1.0 / (avg_variance + 1e-10)
        return importance / np.sum(importance)  # 归一化
    
    def get_support(self, threshold: float = 0.0) -> np.ndarray:
        """
        获取特征选择支持。
        
        Args:
            threshold: 重要性阈值
            
        Returns:
            布尔数组，True表示特征被选中
        """
        if self.gnb_classifier_ is None:
            raise ValueError("Model must be fitted before getting support")
            
        importance = self.get_feature_importance()
        return importance > threshold
    
    def get_params(self, deep: bool = True) -> dict:
        """获取参数。"""
        params = {
            'var_smoothing': self.var_smoothing,
            'random_state': self.random_state
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params) -> 'GaussianNBClassifier':
        """设置参数。"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self
