"""
模型特定的特征选择策略。

为不同模型类型提供最适合的特征选择方法。
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFECV,
    f_classif, mutual_info_classif, chi2
)
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import ElasticNetCV  # 不再使用
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelSpecificFeatureSelector:
    """
    模型特定的特征选择器。
    
    根据模型类型自动选择最适合的特征选择策略。
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        # 定义模型特征选择策略映射
        self.model_strategies = {
            # 线性模型 - 使用内置特征选择
            'lasso': 'built_in',
            'elasticnet': 'built_in',
            
            # 树模型 - 使用基于重要性的选择
            'randomforest': 'importance_based',
            'xgboost': 'importance_based',
            'catboost': 'importance_based',
            
            # 支持向量机 - 使用递归特征消除
            'svm': 'rfe',
            
            # 近邻算法 - 使用统计方法
            'knn': 'statistical',
            
            # 朴素贝叶斯 - 使用统计方法
            'gaussiannb': 'statistical',
            
            # 神经网络 - 使用基于重要性的选择
            'neuralnetwork': 'importance_based',
            'mlp': 'importance_based'
        }
        
        # 特征选择参数配置
        self.selection_params = {
            'importance_based': {
                'threshold': 'median',
                'max_features': 100  # 限制最多保留的特征数
            },
            'rfe': {
                'cv': 3,
                'step': 0.1,
                'min_features': 10
            },
            'statistical': {
                'k': 50,
                'score_func': 'f_classif'
            }
        }
        
        # 自适应阈值配置
        self.adaptive_thresholds = {
            'small_dataset': {'n_samples': 100, 'threshold': 'mean'},  # 小数据集使用均值
            'medium_dataset': {'n_samples': 500, 'threshold': 'median'},  # 中等数据集使用中位数
            'large_dataset': {'n_samples': 1000, 'threshold': '1.5*median'}  # 大数据集使用1.5倍中位数
        }
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: np.ndarray, 
                       model_name: str,
                       **kwargs) -> List[str]:
        """
        为指定模型选择特征。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            model_name: 模型名称
            **kwargs: 额外参数
            
        Returns:
            选中的特征列表
        """
        model_name_lower = model_name.lower()
        
        # 获取策略
        strategy = self.model_strategies.get(model_name_lower, 'statistical')
        
        self.logger.info(f"为模型 {model_name} 使用特征选择策略: {strategy}")
        
        if strategy == 'built_in':
            return self._built_in_selection(X, y, model_name_lower, **kwargs)
        elif strategy == 'importance_based':
            return self._importance_based_selection(X, y, model_name_lower, **kwargs)
        elif strategy == 'rfe':
            return self._rfe_selection(X, y, model_name_lower, **kwargs)
        elif strategy == 'statistical':
            return self._statistical_selection(X, y, model_name_lower, **kwargs)
        else:
            self.logger.warning(f"未知的特征选择策略: {strategy}，使用统计方法")
            return self._statistical_selection(X, y, model_name_lower, **kwargs)
    
    def _built_in_selection(self, 
                           X: pd.DataFrame, 
                           y: np.ndarray, 
                           model_name: str,
                           **kwargs) -> List[str]:
        """
        内置特征选择（如Lasso、ElasticNet）。
        
        这些模型在训练过程中自动进行特征选择。
        """
        # 对于内置特征选择的模型，需要先训练模型获取实际选择的特征
        try:
            if model_name.lower() == 'lasso':
                from ..models.lasso import LassoClassifier
                
                # 尝试不同的C值来找到合适的特征选择
                c_values = [0.01, 0.1, 1.0, 10.0]
                selected_features = []
                
                for c_val in c_values:
                    self.logger.debug(f"尝试Lasso C值: {c_val}")
                    model = LassoClassifier(C=c_val, max_iter=1000)
                    model.fit(X, y)
                    temp_features = model.get_selected_features()
                    
                    # 调试信息
                    coef = model.get_coefficients()
                    non_zero_count = np.sum(coef != 0)
                    self.logger.debug(f"C={c_val}: 非零系数数量: {non_zero_count}")
                    
                    if len(temp_features) > 0:
                        selected_features = temp_features
                        self.logger.info(f"Lasso特征选择成功，C={c_val}，选择了 {len(selected_features)} 个特征")
                        break
                
                # 如果所有C值都返回空集，回退为按方差选择Top-K
                if len(selected_features) == 0:
                    self.logger.warning("所有C值都返回空集，回退为按方差选择Top-50特征")
                    k = min(50, X.shape[1])
                    var = X.var(axis=0)
                    selected_features = var.sort_values(ascending=False).head(k).index.tolist()
                
                self.logger.info(f"{model_name}: 最终选择了 {len(selected_features)} 个特征")
                return selected_features
            elif model_name.lower() == 'elasticnet':
                from ..models.elastic_net import ElasticNetClassifier
                
                # 尝试不同的alpha和l1_ratio组合来找到合适的特征选择
                alpha_values = [0.01, 0.1, 1.0, 10.0]
                l1_ratios = [0.5, 0.7, 0.9]  # 更倾向于L1正则化
                selected_features = []
                
                for alpha in alpha_values:
                    for l1_ratio in l1_ratios:
                        self.logger.debug(f"尝试ElasticNet alpha={alpha}, l1_ratio={l1_ratio}")
                        model = ElasticNetClassifier(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            max_iter=1000
                        )
                        model.fit(X, y)
                        temp_features = model.get_selected_features()
                        
                        # 调试信息
                        coef = model.get_coefficients()
                        non_zero_count = np.sum(coef != 0)
                        self.logger.debug(f"alpha={alpha}, l1_ratio={l1_ratio}: 非零系数数量: {non_zero_count}")
                        
                        if len(temp_features) > 0:
                            selected_features = temp_features
                            self.logger.info(f"ElasticNet特征选择成功，alpha={alpha}, l1_ratio={l1_ratio}，选择了 {len(selected_features)} 个特征")
                            break
                    if len(selected_features) > 0:
                        break
                
                # 如果所有参数组合都返回空集，回退为按方差选择Top-K
                if len(selected_features) == 0:
                    self.logger.warning("所有ElasticNet参数组合都返回空集，回退为按方差选择Top-50特征")
                    k = min(50, X.shape[1])
                    var = X.var(axis=0)
                    selected_features = var.sort_values(ascending=False).head(k).index.tolist()
                
                self.logger.info(f"{model_name}: 最终选择了 {len(selected_features)} 个特征")
                return selected_features
            else:
                # 其他模型返回所有特征
                return X.columns.tolist()
        except Exception as e:
            self.logger.warning(f"内置特征选择失败: {e}，返回所有特征")
            return X.columns.tolist()
    
    def _importance_based_selection(self, 
                                   X: pd.DataFrame, 
                                   y: np.ndarray, 
                                   model_name: str,
                                   **kwargs) -> List[str]:
        """
        基于重要性的特征选择。
        
        适用于树模型和神经网络。
        """
        params = self.selection_params['importance_based']
        threshold = kwargs.get('threshold', params['threshold'])
        max_features = kwargs.get('max_features', params['max_features'])
        
        # 自适应阈值选择
        n_samples = X.shape[0]
        if n_samples <= self.adaptive_thresholds['small_dataset']['n_samples']:
            threshold = 'median'
            self.logger.info(f"小数据集({n_samples}样本)，使用中位数阈值")
        elif n_samples <= self.adaptive_thresholds['medium_dataset']['n_samples']:
            threshold = '1.5*median'
            self.logger.info(f"中等数据集({n_samples}样本)，使用1.5倍中位数阈值")
        else:
            threshold = '1.5*median'
            self.logger.info(f"大数据集({n_samples}样本)，使用1.5倍中位数阈值")
        
        # 创建基础模型用于特征重要性计算
        if model_name in ['randomforest', 'rf']:
            base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        elif model_name in ['xgboost']:
            try:
                from xgboost import XGBClassifier
                base_model = XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
            except ImportError:
                self.logger.warning("XGBoost not available, using RandomForest")
                base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        elif model_name in ['catboost']:
            try:
                from catboost import CatBoostClassifier
                base_model = CatBoostClassifier(iterations=50, random_seed=42, verbose=False)
            except ImportError:
                self.logger.warning("CatBoost not available, using RandomForest")
                base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            # 默认使用随机森林
            base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # 使用SelectFromModel进行特征选择
        selector = SelectFromModel(
            estimator=base_model,
            threshold=threshold,
            max_features=max_features
        )
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # 计算特征重要性的统计信息
        if hasattr(selector.estimator_, 'feature_importances_'):
            importances = selector.estimator_.feature_importances_
            self.logger.info(f"特征重要性统计: 均值={importances.mean():.4f}, 中位数={np.median(importances):.4f}, 最大值={importances.max():.4f}")
            self.logger.info(f"选择的特征重要性范围: {importances[selector.get_support()].min():.4f} - {importances[selector.get_support()].max():.4f}")
        
        self.logger.info(f"基于重要性选择了 {len(selected_features)} 个特征 (阈值: {threshold})")
        return selected_features
    
    def _rfe_selection(self, 
                      X: pd.DataFrame, 
                      y: np.ndarray, 
                      model_name: str,
                      **kwargs) -> List[str]:
        """
        递归特征消除（RFE）。
        
        适用于支持向量机等模型。
        """
        params = self.selection_params['rfe']
        cv = kwargs.get('cv', params['cv'])
        step = kwargs.get('step', params['step'])
        min_features = kwargs.get('min_features', params['min_features'])
        
        # 创建基础模型
        if model_name in ['svm', 'svc']:
            base_model = SVC(kernel='linear', probability=True, random_state=42)
        else:
            # 默认使用线性SVM
            base_model = SVC(kernel='linear', probability=True, random_state=42)
        
        # 使用RFECV进行特征选择
        selector = RFECV(
            estimator=base_model,
            cv=cv,
            step=step,
            min_features_to_select=min_features,
            scoring='roc_auc'
        )
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.logger.info(f"RFE选择了 {len(selected_features)} 个特征")
        return selected_features
    
    def _statistical_selection(self, 
                              X: pd.DataFrame, 
                              y: np.ndarray, 
                              model_name: str,
                              **kwargs) -> List[str]:
        """
        统计方法特征选择。
        
        适用于KNN、朴素贝叶斯等模型。
        """
        params = self.selection_params['statistical']
        k = kwargs.get('k', params['k'])
        score_func = kwargs.get('score_func', params['score_func'])
        
        # 选择评分函数
        if score_func == 'f_classif':
            score_function = f_classif
        elif score_func == 'mutual_info':
            score_function = mutual_info_classif
        elif score_func == 'chi2':
            score_function = chi2
        else:
            score_function = f_classif
        
        # 使用SelectKBest进行特征选择
        k = min(k, X.shape[1])  # 确保k不超过特征数
        selector = SelectKBest(score_func=score_function, k=k)
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.logger.info(f"统计方法选择了 {len(selected_features)} 个特征")
        return selected_features
    
    def get_strategy_for_model(self, model_name: str) -> str:
        """
        获取模型的特征选择策略。
        
        Args:
            model_name: 模型名称
            
        Returns:
            特征选择策略名称
        """
        return self.model_strategies.get(model_name.lower(), 'statistical')
    
    def add_model_strategy(self, model_name: str, strategy: str) -> None:
        """
        添加新的模型策略映射。
        
        Args:
            model_name: 模型名称
            strategy: 特征选择策略
        """
        self.model_strategies[model_name.lower()] = strategy
        self.logger.info(f"为模型 {model_name} 添加了策略 {strategy}")
    
    def update_selection_params(self, strategy: str, params: Dict[str, Any]) -> None:
        """
        更新特征选择参数。
        
        Args:
            strategy: 策略名称
            params: 参数字典
        """
        if strategy in self.selection_params:
            self.selection_params[strategy].update(params)
            self.logger.info(f"更新了策略 {strategy} 的参数")
        else:
            self.selection_params[strategy] = params
            self.logger.info(f"添加了新策略 {strategy} 的参数")
