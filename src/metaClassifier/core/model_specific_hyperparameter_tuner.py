"""
模型特定的超参数调优策略。

为不同模型类型提供最适合的超参数搜索空间和调优方法。
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.linear_model import ElasticNetCV  # 不再使用
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelSpecificHyperparameterTuner:
    """
    模型特定的超参数调优器。
    
    根据模型类型自动选择最适合的超参数搜索空间和调优方法。
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        # 定义模型超参数搜索空间
        self.param_grids = {
            # 线性模型
            'lasso': {
                # LogisticRegression with L1 penalty 参数
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],  # 增加更大的C值
                'max_iter': [1000, 2000, 5000],  # 最大迭代次数
                'tol': [1e-4, 1e-6, 1e-8],  # 收敛容差
                'n_jobs': [1, 2, 4],  # 并行数
                'random_state': [42, 123, 456],  # 随机种子
            },
            'elasticnet': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [1000, 2000]
            },
            
            # 树模型
            'randomforest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'catboost': {
                'iterations': [50, 100, 200],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            },
            
            # 支持向量机
            'svm': {
                'C': [0.1, 1, 10, 100, 1000],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'degree': [2, 3, 4, 5]
            },
            
            # 近邻算法
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'p': [1, 2, 3]
            },
            
            # 朴素贝叶斯
            'gaussiannb': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            
            # 神经网络
            'neuralnetwork': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [200, 500, 1000]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [200, 500]
            }
        }
        
        # 定义调优方法
        self.tuning_methods = {
            'lasso': 'grid_search',  # 使用网格搜索
            'elasticnet': 'cv_based',  # 使用ElasticNetCV
            'randomforest': 'random_search',
            'xgboost': 'grid_search',
            'catboost': 'grid_search',
            'svm': 'random_search',
            'knn': 'grid_search',
            'gaussiannb': 'grid_search',
            'neuralnetwork': 'random_search',
            'mlp': 'random_search'
        }
        
        # 搜索参数
        self.search_params = {
            'grid_search': {
                'cv': 3,
                'scoring': 'roc_auc',
                'n_jobs': -1,
                'verbose': 0
            },
            'random_search': {
                'cv': 3,
                'scoring': 'roc_auc',
                'n_jobs': -1,
                'verbose': 0,
                'n_iter': 20
            },
            'cv_based': {
                'cv': 3,
                'scoring': 'roc_auc',
                'n_jobs': -1
            }
        }
    
    def tune_hyperparameters(self, 
                            X: pd.DataFrame, 
                            y: np.ndarray, 
                            model_name: str,
                            **kwargs) -> Dict[str, Any]:
        """
        为指定模型调优超参数。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            model_name: 模型名称
            **kwargs: 额外参数
            
        Returns:
            最佳超参数字典
        """
        model_name_lower = model_name.lower()
        
        # 获取调优方法
        method = self.tuning_methods.get(model_name_lower, 'grid_search')
        
        self.logger.info(f"为模型 {model_name} 使用超参数调优方法: {method}")
        
        if method == 'cv_based':
            return self._cv_based_tuning(X, y, model_name_lower, **kwargs)
        elif method == 'grid_search':
            return self._grid_search_tuning(X, y, model_name_lower, **kwargs)
        elif method == 'random_search':
            return self._random_search_tuning(X, y, model_name_lower, **kwargs)
        else:
            self.logger.warning(f"未知的调优方法: {method}，使用网格搜索")
            return self._grid_search_tuning(X, y, model_name_lower, **kwargs)

    # ========== 新增：仅提供每个模型的参数搜索空间（供通用调参器使用） ==========
    def get_adaptive_param_grid(self, model_name: str, data_shape: tuple, n_samples: int, n_features: int, class_balance: Optional[float] = None) -> Dict[str, Any]:
        """
        根据数据特征自适应调整参数搜索空间。
        
        Args:
            model_name: 模型名称
            data_shape: 数据形状 (n_samples, n_features)
            n_samples: 样本数量
            n_features: 特征数量
            
        Returns:
            自适应调整后的参数网格
        """
        base_grid = self.get_param_grid(model_name)
        name = model_name.lower()
        
        # 根据数据特征调整参数
        if n_features > 1000:  # 高维数据
            if name == 'lasso':
                # 高维数据需要更强的正则化
                base_grid['alpha'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
            elif name == 'randomforest':
                # 高维数据减少特征采样
                base_grid['max_features'] = ['sqrt', 'log2', 0.05, 0.1]
        
        if n_samples < 100:  # 小样本数据
            if name == 'randomforest':
                # 小样本数据减少树的数量和深度
                base_grid['n_estimators'] = [50, 100, 200]
                base_grid['max_depth'] = [5, 10, 15]
                base_grid['min_samples_split'] = [5, 10, 20]
        
        # 处理类别不平衡
        if class_balance is not None and class_balance < 0.3:  # 严重不平衡
            if name == 'lasso':
                # Lasso不支持class_weight，通过调整alpha来处理不平衡
                # 不平衡数据使用更强的正则化
                base_grid['alpha'] = [0.001, 0.01, 0.1, 1.0, 10.0]
            elif name == 'randomforest':
                # 添加class_weight参数
                base_grid['class_weight'] = ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}]
                # 调整采样策略
                base_grid['max_samples'] = [0.8, 0.9, 1.0]
            elif name == 'xgboost':
                # 添加scale_pos_weight参数
                pos_weight = int(1 / class_balance) if class_balance > 0 else 1
                base_grid['scale_pos_weight'] = [1, pos_weight, pos_weight * 2]
        
        return base_grid
    
    def get_param_grid(self, model_name: str) -> Dict[str, Any]:
        """
        返回指定模型的超参数搜索空间（离散网格/候选集）。
        仅定义“搜什么”，不执行搜索流程。
        """
        name = (model_name or "").lower()
        if name == 'lasso':
            # 针对宏基因组数据优化的Lasso参数空间
            return {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],  # 更宽的alpha范围
                'max_iter': [2000, 5000, 10000],  # 增加迭代次数
                'tol': [1e-5, 1e-4, 1e-3],  # 调整容差
                'random_state': [42]
            }
        if name == 'randomforest':
            # 针对宏基因组数据优化的RandomForest参数空间
            return {
                'n_estimators': [100, 200, 500, 1000],  # 增加树的数量
                'max_depth': [None, 10, 20, 30],  # 更深的树
                'min_samples_split': [2, 5, 10, 20],  # 适应高维数据
                'min_samples_leaf': [1, 2, 4, 8],  # 防止过拟合
                'max_features': ['sqrt', 'log2', 0.1, 0.2],  # 特征采样策略
                'bootstrap': [True, False],  # 是否使用bootstrap
                'random_state': [42]
            }
        if name == 'xgboost':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        if name == 'catboost':
            return {
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1],
                'iterations': [200, 500]
            }
        if name == 'svm':
            # kernel 与参数联动：此处给出常见组合；执行器可按需过滤非法组合
            return {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        if name == 'neuralnetwork':
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [1e-4, 1e-3, 1e-2],
                'learning_rate_init': [1e-3, 1e-2],
                'max_iter': [200, 400]
            }
        # 默认空字典：调用方将直接使用模型默认参数
        return {}
    
    def get_bayesian_param_grid(self, model_name: str) -> Dict[str, Any]:
        """
        返回指定模型的贝叶斯优化参数搜索空间。
        为贝叶斯优化提供连续参数范围，而不是离散网格。
        """
        name = (model_name or "").lower()
        if name == 'lasso':
            # Lasso的贝叶斯参数范围
            return {
                'alpha': [0.001, 10.0],  # 正则化强度，对数尺度，控制稀疏性
                'max_iter': [1000, 5000],  # 最大迭代次数，整数范围
                'tol': [1e-6, 1e-3],  # 收敛容差，对数尺度
            }
        if name == 'randomforest':
            return {
                'n_estimators': [50, 500],  # 整数范围
                'max_depth': [5, 30],  # 整数范围
                'min_samples_split': [2, 20],  # 整数范围
                'min_samples_leaf': [1, 10],  # 整数范围
                'max_features': [0.1, 1.0],  # 浮点数范围，表示特征采样比例
            }
        if name == 'xgboost':
            return {
                'n_estimators': [50, 500],  # 整数范围
                'max_depth': [3, 15],  # 整数范围
                'learning_rate': [0.01, 0.3],  # 对数尺度范围
                'subsample': [0.6, 1.0],  # 线性范围
            }
        if name == 'catboost':
            return {
                'depth': [4, 12],  # 整数范围
                'learning_rate': [0.01, 0.3],  # 对数尺度范围
                'iterations': [100, 1000],  # 整数范围
                'l2_leaf_reg': [1, 20],  # 整数范围
            }
        if name == 'svm':
            return {
                'C': [0.1, 1000],  # 对数尺度范围
                'gamma': [1e-4, 1],  # 对数尺度范围
            }
        if name == 'neuralnetwork':
            return {
                'alpha': [1e-5, 1e-1],  # 对数尺度范围
                'learning_rate_init': [1e-4, 1e-1],  # 对数尺度范围
                'max_iter': [200, 1000],  # 整数范围
            }
        if name == 'logistic':
            # 一般逻辑回归的贝叶斯参数范围
            return {
                'C': [0.001, 100.0],  # 正则化强度，对数尺度
                'max_iter': [1000, 5000],  # 最大迭代次数，整数范围
                'tol': [1e-6, 1e-3],  # 收敛容差，对数尺度
            }
        if name == 'elasticnet':
            # ElasticNet的贝叶斯参数范围
            return {
                'alpha': [0.001, 10.0],  # 正则化强度，对数尺度，控制稀疏性
                'l1_ratio': [0.1, 0.95],  # L1正则化比例，线性范围，控制稀疏性
                'max_iter': [1000, 5000],  # 最大迭代次数，整数范围
                'tol': [1e-6, 1e-3],  # 收敛容差，对数尺度
            }
        # 默认返回空字典
        return {}
    
    def _cv_based_tuning(self, 
                        X: pd.DataFrame, 
                        y: np.ndarray, 
                        model_name: str,
                        **kwargs) -> Dict[str, Any]:
        """
        基于交叉验证的调优（如ElasticNetCV）。
        """
        if model_name == 'lasso':
            # lasso 现在使用 LogisticRegression，回退到网格搜索
            return self._grid_search_tuning(X, y, model_name, **kwargs)
            
        elif model_name == 'elasticnet':
            # elasticnet 现在使用 LogisticRegression，回退到网格搜索
            return self._grid_search_tuning(X, y, model_name, **kwargs)
        else:
            # 回退到网格搜索
            return self._grid_search_tuning(X, y, model_name, **kwargs)
    
    def _grid_search_tuning(self, 
                           X: pd.DataFrame, 
                           y: np.ndarray, 
                           model_name: str,
                           **kwargs) -> Dict[str, Any]:
        """
        网格搜索调优。
        """
        # 获取参数网格
        param_grid = self.param_grids.get(model_name, {})
        if not param_grid:
            self.logger.warning(f"没有为模型 {model_name} 定义参数网格")
            return {}
        
        # 获取搜索参数
        search_params = self.search_params['grid_search'].copy()
        search_params.update(kwargs)
        
        # 创建基础模型
        base_model = self._create_base_model(model_name)
        
        # 执行网格搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            **search_params
        )
        
        grid_search.fit(X, y)
        
        self.logger.info(f"网格搜索最佳参数: {grid_search.best_params_}")
        self.logger.info(f"最佳得分: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def _random_search_tuning(self, 
                             X: pd.DataFrame, 
                             y: np.ndarray, 
                             model_name: str,
                             **kwargs) -> Dict[str, Any]:
        """
        随机搜索调优。
        """
        # 获取参数网格
        param_grid = self.param_grids.get(model_name, {})
        if not param_grid:
            self.logger.warning(f"没有为模型 {model_name} 定义参数网格")
            return {}
        
        # 获取搜索参数
        search_params = self.search_params['random_search'].copy()
        search_params.update(kwargs)
        
        # 创建基础模型
        base_model = self._create_base_model(model_name)
        
        # 执行随机搜索
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            **search_params
        )
        
        random_search.fit(X, y)
        
        self.logger.info(f"随机搜索最佳参数: {random_search.best_params_}")
        self.logger.info(f"最佳得分: {random_search.best_score_:.4f}")
        
        return random_search.best_params_
    
    def _create_base_model(self, model_name: str):
        """
        创建基础模型实例。
        """
        if model_name == 'randomforest':
            return RandomForestClassifier(random_state=42)
        elif model_name == 'xgboost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(random_state=42, verbosity=0)
            except ImportError:
                self.logger.warning("XGBoost not available, using RandomForest")
                return RandomForestClassifier(random_state=42)
        elif model_name == 'catboost':
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(random_seed=42, verbose=False)
            except ImportError:
                self.logger.warning("CatBoost not available, using RandomForest")
                return RandomForestClassifier(random_state=42)
        elif model_name == 'svm':
            return SVC(probability=True, random_state=42)
        elif model_name == 'knn':
            return KNeighborsClassifier()
        elif model_name == 'gaussiannb':
            return GaussianNB()
        elif model_name in ['neuralnetwork', 'mlp']:
            return MLPClassifier(random_state=42, max_iter=200)
        else:
            # 默认使用随机森林
            return RandomForestClassifier(random_state=42)
    
    def get_param_grid(self, model_name: str) -> Dict[str, List[Any]]:
        """
        获取模型的参数网格。
        
        Args:
            model_name: 模型名称
            
        Returns:
            参数网格字典
        """
        return self.param_grids.get(model_name.lower(), {})
    
    def get_tuning_method(self, model_name: str) -> str:
        """
        获取模型的调优方法。
        
        Args:
            model_name: 模型名称
            
        Returns:
            调优方法名称
        """
        return self.tuning_methods.get(model_name.lower(), 'grid_search')
    
    def add_model_config(self, 
                        model_name: str, 
                        param_grid: Dict[str, List[Any]], 
                        tuning_method: str = 'grid_search') -> None:
        """
        添加新模型的配置。
        
        Args:
            model_name: 模型名称
            param_grid: 参数网格
            tuning_method: 调优方法
        """
        self.param_grids[model_name.lower()] = param_grid
        self.tuning_methods[model_name.lower()] = tuning_method
        self.logger.info(f"为模型 {model_name} 添加了配置")
    
    def update_param_grid(self, model_name: str, param_grid: Dict[str, List[Any]]) -> None:
        """
        更新模型的参数网格。
        
        Args:
            model_name: 模型名称
            param_grid: 新的参数网格
        """
        if model_name.lower() in self.param_grids:
            self.param_grids[model_name.lower()].update(param_grid)
            self.logger.info(f"更新了模型 {model_name} 的参数网格")
        else:
            self.logger.warning(f"模型 {model_name} 不存在，无法更新")
