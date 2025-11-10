"""
ElasticNet分类器实现。

ElasticNet结合了L1和L2正则化，适合高维稀疏数据。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from typing import Optional, Union

from .base_model import BaseModel


class ElasticNetClassifier(BaseModel):
    """
    ElasticNet分类器。
    
    使用LogisticRegression with elasticnet penalty进行二分类。
    """
    
    # 明确指定分类器类型，满足新版本scikit-learn的要求
    _estimator_type = "classifier"
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, 
                 max_iter: int = 1000, tol: float = 1e-4, 
                 random_state: Optional[int] = None, 
                 threshold: float = 0.5, **kwargs):
        """
        初始化ElasticNet分类器。
        
        Args:
            alpha: 正则化强度
            l1_ratio: L1正则化比例 (0-1之间，0为纯L2，1为纯L1)
            max_iter: 最大迭代次数
            tol: 收敛容差
            random_state: 随机种子
            threshold: 分类阈值
        """
        super().__init__("ElasticNet", **kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.threshold = threshold
        
        self.logistic_regression_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'ElasticNetClassifier':
        """
        训练ElasticNet分类器。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            
        Returns:
            self
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        
        # 检查二分类
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("ElasticNetClassifier only supports binary classification")
        
        # 创建LogisticRegression with elasticnet penalty
        self.logistic_regression_ = LogisticRegression(
            penalty='elasticnet',
            C=1.0/self.alpha,  # sklearn使用C=1/alpha
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            solver='saga'  # saga solver支持elasticnet
        )
        
        # 训练
        self.logistic_regression_.fit(X, y)
        
        # 调用BaseModel的fit方法
        super().fit(X, y)
            
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测分类结果。
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测标签
        """
        if self.logistic_regression_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=True)
        return self.logistic_regression_.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测概率。
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率 [P(class0), P(class1)]
        """
        if self.logistic_regression_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=True)
        return self.logistic_regression_.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性（系数绝对值）。
        
        Returns:
            特征重要性数组
        """
        if self.logistic_regression_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
            
        return np.abs(self.logistic_regression_.coef_[0])
    
    def get_support(self) -> np.ndarray:
        """
        获取特征选择支持（非零系数）。
        
        Returns:
            布尔数组，True表示特征被选中
        """
        if self.logistic_regression_ is None:
            raise ValueError("Model must be fitted before getting support")
            
        return self.logistic_regression_.coef_[0] != 0
    
    def get_selected_features(self) -> list:
        """
        获取选中的特征名称。
        
        Returns:
            选中特征的名称列表
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting selected features")
            
        try:
            # 使用原始系数而不是绝对值来判断特征是否被选择
            coef = self.logistic_regression_.coef_[0]
            selected_mask = coef != 0
            return [self.feature_names_[i] for i in range(len(self.feature_names_)) if selected_mask[i]]
        except Exception as e:
            # 使用print而不是logger，因为可能没有logger属性
            print(f"警告: 获取选中特征时出错: {e}")
            return []
    
    def get_coefficients(self) -> np.ndarray:
        """
        获取ElasticNet系数。
        
        Returns:
            系数数组
        """
        if self.logistic_regression_ is None:
            raise ValueError("Model must be fitted before getting coefficients")
            
        return self.logistic_regression_.coef_[0]
    
    def get_params(self, deep: bool = True) -> dict:
        """获取参数。"""
        return {
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'threshold': self.threshold
        }
    
    def set_params(self, **params) -> 'ElasticNetClassifier':
        """设置参数。"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
