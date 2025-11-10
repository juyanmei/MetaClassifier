"""
SVM分类器实现。

支持多种核函数的支持向量机分类器。
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC as _SVC
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class SVMClassifier(BaseEstimator, ClassifierMixin):
    """
    SVM分类器包装器。
    
    提供与sklearn兼容的接口，并自动进行特征标准化。
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', 
                 gamma: Union[str, float] = 'scale', degree: int = 3,
                 coef0: float = 0.0, shrinking: bool = True,
                 probability: bool = True, random_state: Optional[int] = None,
                 **kwargs):
        """
        初始化SVM分类器。
        
        Args:
            C: 正则化参数
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: 核函数参数
            degree: 多项式核的度数
            coef0: 核函数中的独立项
            shrinking: 是否使用收缩启发式
            probability: 是否启用概率估计
            random_state: 随机种子
            **kwargs: 其他SVM参数
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.random_state = random_state
        
        # 存储其他参数
        self.kwargs = kwargs
        
        self.svm_classifier_ = None
        self.scaler_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'SVMClassifier':
        """
        训练SVM分类器。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            
        Returns:
            self
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # 获取类别信息
        self.classes_ = unique_labels(y)
        
        # 标准化特征
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # 创建SVM分类器
        self.svm_classifier_ = _SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            random_state=self.random_state,
            **self.kwargs
        )
        
        # 训练
        self.svm_classifier_.fit(X_scaled, y)
        
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
        if self.svm_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        X_scaled = self.scaler_.transform(X)
        return self.svm_classifier_.predict(X_scaled)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测概率。
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率
        """
        if self.svm_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        X_scaled = self.scaler_.transform(X)
        return self.svm_classifier_.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性。
        
        注意：只有线性核的SVM才有特征重要性。
        
        Returns:
            特征重要性数组
        """
        if self.svm_classifier_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
            
        if self.kernel != 'linear':
            # 对于非线性核，返回零向量
            return np.zeros(self.n_features_in_)
        
        # 对于线性核，返回系数的绝对值
        return np.abs(self.svm_classifier_.coef_[0])
    
    def get_support(self, threshold: float = 0.0) -> np.ndarray:
        """
        获取特征选择支持。
        
        Args:
            threshold: 重要性阈值
            
        Returns:
            布尔数组，True表示特征被选中
        """
        if self.svm_classifier_ is None:
            raise ValueError("Model must be fitted before getting support")
            
        if self.kernel != 'linear':
            # 对于非线性核，返回全True（所有特征都被使用）
            return np.ones(self.n_features_in_, dtype=bool)
        
        importance = self.get_feature_importance()
        return importance > threshold
    
    def get_params(self, deep: bool = True) -> dict:
        """获取参数。"""
        params = {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'shrinking': self.shrinking,
            'probability': self.probability,
            'random_state': self.random_state
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params) -> 'SVMClassifier':
        """设置参数。"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self
