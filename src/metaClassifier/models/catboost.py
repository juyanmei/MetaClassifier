"""
CatBoost分类器实现。

CatBoost是一个高效的梯度提升算法，对类别特征处理友好。
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels

try:
    from catboost import CatBoostClassifier as _CatBoostClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    _CatBoostClassifier = None

from .base_model import BaseModel


class CatBoostClassifier(BaseModel):
    """
    CatBoost分类器包装器。
    
    提供与sklearn兼容的接口。
    """
    
    # 明确指定分类器类型，满足新版本scikit-learn的要求
    _estimator_type = "classifier"
    
    def __sklearn_tags__(self):
        """返回sklearn标签，确保被识别为分类器。"""
        from sklearn.utils._tags import Tags, TargetTags, ClassifierTags, InputTags
        
        return Tags(
            estimator_type='classifier',
            target_tags=TargetTags(
                required=True,
                one_d_labels=True,
                two_d_labels=False,
                positive_only=False,
                multi_output=False,
                single_output=True
            ),
            transformer_tags=None,
            classifier_tags=ClassifierTags(
                poor_score=False,
                multi_class=True,
                multi_label=False
            ),
            regressor_tags=None,
            array_api_support=False,
            no_validation=False,
            non_deterministic=False,
            requires_fit=True,
            _skip_test=False,
            input_tags=InputTags(
                one_d_array=False,
                two_d_array=True,
                three_d_array=False,
                sparse=False,
                categorical=False,
                string=False,
                dict=False,
                positive_only=False,
                allow_nan=False,
                pairwise=False
            )
        )
    
    def __init__(self, iterations: int = 100, depth: int = 6, 
                 learning_rate: float = 0.1, l2_leaf_reg: float = 3.0,
                 random_seed: Optional[int] = None, verbose: bool = False,
                 **kwargs):
        """
        初始化CatBoost分类器。
        
        Args:
            iterations: 迭代次数
            depth: 树的最大深度
            learning_rate: 学习率
            l2_leaf_reg: L2正则化参数
            random_seed: 随机种子
            verbose: 是否显示训练过程
            **kwargs: 其他CatBoost参数
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available. Install with: pip install catboost")
            
        super().__init__("CatBoost", **kwargs)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose
        
        # 存储其他参数
        self.kwargs = kwargs
        
        self.catboost_classifier_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'CatBoostClassifier':
        """
        训练CatBoost分类器。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            
        Returns:
            self
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # 获取类别信息
        self.classes_ = unique_labels(y)
        
        # 创建CatBoost分类器
        self.catboost_classifier_ = _CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=self.verbose,
            loss_function='Logloss',  # 明确指定分类损失函数
            **self.kwargs
        )
        
        # 训练
        self.catboost_classifier_.fit(X, y)
        
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
        if self.catboost_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        return self.catboost_classifier_.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测概率。
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率
        """
        if self.catboost_classifier_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = check_array(X, accept_sparse=False)
        return self.catboost_classifier_.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性。
        
        Returns:
            特征重要性数组
        """
        if self.catboost_classifier_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
            
        return self.catboost_classifier_.feature_importances_
    
    def get_support(self, threshold: float = 0.0) -> np.ndarray:
        """
        获取特征选择支持。
        
        Args:
            threshold: 重要性阈值
            
        Returns:
            布尔数组，True表示特征被选中
        """
        if self.catboost_classifier_ is None:
            raise ValueError("Model must be fitted before getting support")
            
        importance = self.get_feature_importance()
        return importance > threshold
    
    def get_params(self, deep: bool = True) -> dict:
        """获取参数。"""
        params = {
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'l2_leaf_reg': self.l2_leaf_reg,
            'random_seed': self.random_seed,
            'verbose': self.verbose
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params) -> 'CatBoostClassifier':
        """设置参数。"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self
    
