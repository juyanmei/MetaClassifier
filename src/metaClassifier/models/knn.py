"""
KNN分类器实现。

K近邻分类器，适合小样本数据。
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    KNN分类器包装器。
    
    提供与sklearn兼容的接口，并自动进行特征标准化。
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform',
                 algorithm: str = 'auto', leaf_size: int = 30,
                 p: int = 2, metric: str = 'minkowski',
                 random_state: Optional[int] = None, **kwargs):
        """
        初始化KNN分类器。
        
        Args:
            n_neighbors: 邻居数量
            weights: 权重函数 ('uniform', 'distance')
            algorithm: 算法类型 ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size: 叶子节点大小
            p: 距离度量的幂参数
            metric: 距离度量
            random_state: 随机种子
            **kwargs: 其他KNN参数
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.random_state = random_state
        
        # 存储其他参数
        self.kwargs = kwargs
        
        self.knn_classifier_ = None
        self.scaler_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'KNNClassifier':
        """
        训练KNN分类器。
        
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
        
        # 创建KNN分类器
        self.knn_classifier_ = _KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            **self.kwargs
        )
        
        # 训练
        self.knn_classifier_.fit(X_scaled, y)
        
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
        if self.knn_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        X_scaled = self.scaler_.transform(X)
        return self.knn_classifier_.predict(X_scaled)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测概率。
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率
        """
        if self.knn_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        X_scaled = self.scaler_.transform(X)
        return self.knn_classifier_.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性。
        
        注意：KNN没有直接的特征重要性，返回均匀分布。
        
        Returns:
            特征重要性数组
        """
        if self.knn_classifier_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
            
        # KNN没有特征重要性，返回均匀分布
        return np.ones(self.n_features_in_) / self.n_features_in_
    
    def get_support(self, threshold: float = 0.0) -> np.ndarray:
        """
        获取特征选择支持。
        
        Args:
            threshold: 重要性阈值
            
        Returns:
            布尔数组，True表示特征被选中
        """
        if self.knn_classifier_ is None:
            raise ValueError("Model must be fitted before getting support")
            
        # KNN使用所有特征
        return np.ones(self.n_features_in_, dtype=bool)
    
    def get_params(self, deep: bool = True) -> dict:
        """获取参数。"""
        params = {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'p': self.p,
            'metric': self.metric,
            'random_state': self.random_state
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params) -> 'KNNClassifier':
        """设置参数。"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self
