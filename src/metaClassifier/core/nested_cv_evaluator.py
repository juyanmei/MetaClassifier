"""
嵌套交叉验证评估器 for metaClassifier v1.0.

这个模块实现了完整的嵌套CV逻辑：
- 外层CV：无偏估计模型泛化性能
- 内层CV：特征选择 + 超参数调优
- 输出：外层平均性能指标 + 共识特征集
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging
from collections import defaultdict, Counter

from .base import BaseEvaluator, CVConfig, CVStrategy, BaseModel
from ..utils.logger import get_logger
from .feature_selector import FeatureSelectorFactory
from .hyperparameter_tuner import HyperparameterTunerFactory
from .model_specific_feature_selector import ModelSpecificFeatureSelector
from .model_specific_hyperparameter_tuner import ModelSpecificHyperparameterTuner


class NestedCVEvaluator(BaseEvaluator):
    """
    嵌套交叉验证评估器。
    
    实现完整的嵌套CV逻辑：
    1. 外层CV：无偏估计模型泛化性能
    2. 内层CV：特征选择 + 超参数调优
    3. 输出：外层平均性能指标 + 共识特征集
    """
    
    def __init__(self, config: CVConfig, feature_selection_config: Optional['FeatureSelectionConfig'] = None, experiment_config: Optional['ExperimentConfig'] = None):
        super().__init__(config)
        self.logger = get_logger("NestedCVEvaluator")
        
        # 验证配置
        self._validate_configs(config, feature_selection_config, experiment_config)
        
        # 特征选择配置
        self.feature_selection_config = feature_selection_config
        # 实验配置（用于获取search_method等参数）
        self.experiment_config = experiment_config
        
        # 初始化工厂
        self.feature_selector_factory = FeatureSelectorFactory()
        self.hyperparameter_tuner_factory = HyperparameterTunerFactory()
        self.model_specific_selector = ModelSpecificFeatureSelector()
        self.model_specific_tuner = ModelSpecificHyperparameterTuner()
        
        # 结果存储
        self.outer_fold_results_ = []
        self.consensus_features_ = {}
        self.performance_metrics_ = {}
    
    def _validate_configs(self, config: CVConfig, feature_selection_config, experiment_config):
        """验证配置参数的有效性"""
        self.logger.debug("验证配置参数...")
        
        # 验证CV配置
        if config.outer_folds < 2:
            raise ValueError(f"外层CV折数必须大于等于2，当前值: {config.outer_folds}")
        if config.inner_folds < 2:
            raise ValueError(f"内层CV折数必须大于等于2，当前值: {config.inner_folds}")
        if config.n_repeats < 1:
            raise ValueError(f"重复次数必须大于等于1，当前值: {config.n_repeats}")
        if config.n_jobs < 1:
            raise ValueError(f"并行作业数必须大于等于1，当前值: {config.n_jobs}")
        
        # 验证特征选择配置
        if feature_selection_config:
            if not 0 <= feature_selection_config.threshold <= 1:
                raise ValueError(f"特征选择阈值必须在0-1之间，当前值: {feature_selection_config.threshold}")
            if hasattr(feature_selection_config, 'search_method'):
                valid_methods = ['grid', 'random', 'bayes']
                if feature_selection_config.search_method not in valid_methods:
                    raise ValueError(f"搜索方法必须是 {valid_methods} 之一，当前值: {feature_selection_config.search_method}")
        
        # 验证实验配置
        if experiment_config:
            if hasattr(experiment_config, 'search_method'):
                valid_methods = ['grid', 'random', 'bayes']
                if experiment_config.search_method not in valid_methods:
                    raise ValueError(f"实验搜索方法必须是 {valid_methods} 之一，当前值: {experiment_config.search_method}")
        
        self.logger.debug("配置验证通过")
        
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        models: Dict[str, BaseModel],
        cohort_info: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        执行嵌套交叉验证评估。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            models: 模型字典
            cohort_info: 队列信息（用于LOCO）
            
        Returns:
            嵌套CV评估结果
        """
        self.logger.info("开始嵌套交叉验证评估...")
        self.logger.info(f"数据维度: {X.shape}")
        self.logger.info(f"模型数量: {len(models)}")
        
        # 确定外层CV策略
        outer_cv = self._create_outer_cv(cohort_info)
        
        # 存储结果
        self.outer_fold_results_ = []
        self.consensus_features_ = {}
        self.performance_metrics_ = {}
        
        # 存储自适应参数和类别不平衡信息（用于可重现性）
        self.adaptive_param_info_ = {}
        self.class_balance_info_ = {}
        self.label_mapping_ = None  # 存储标签映射信息
        
        # 外层CV循环
        total_splits = outer_cv.get_n_splits(X, y, groups=cohort_info)
        self.logger.info(f"总分割数: {total_splits}")
        
        for split_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=cohort_info)):
            self.logger.info(f"处理分割 {split_idx + 1}/{total_splits}")
            
            # 正确处理重复CV的索引
            if self.config.n_repeats > 1 and self.config.strategy != CVStrategy.LOCO:
                # 重复CV: split_idx = repeat * n_folds + fold
                fold_idx = split_idx % self.config.outer_folds
                repeat_idx = split_idx // self.config.outer_folds
                self.logger.info(f"--- 重复 {repeat_idx + 1}/{self.config.n_repeats}, 折 {fold_idx + 1}/{self.config.outer_folds} ---")
            elif self.config.strategy == CVStrategy.LOCO:
                # LOCO: 每个分割对应一个队列
                fold_idx = split_idx
                repeat_idx = 0
                self.logger.info(f"--- LOCO 队列 {fold_idx + 1}/{total_splits} ---")
            else:
                # 普通CV
                fold_idx = split_idx
                repeat_idx = 0
                self.logger.info(f"--- 外层折 {fold_idx + 1}/{self.config.outer_folds} ---")
            
            # 分割数据
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 确保样本ID信息被正确保存
            train_sample_ids = X_train.index.tolist()
            test_sample_ids = X_test.index.tolist()
            
            # 执行内层CV优化
            inner_results = self._execute_inner_cv_optimization(
                X_train, y_train, models, fold_idx
            )
            
            # 使用内层CV结果训练最终模型并测试
            outer_fold_result = self._evaluate_outer_fold(
                X_train, X_test, y_train, y_test, 
                models, inner_results, fold_idx, repeat_idx,
                train_sample_ids, test_sample_ids, cohort_info, test_idx
            )
            
            self.outer_fold_results_.append(outer_fold_result)
            
        # 计算共识特征和性能指标
        self._calculate_consensus_features()
        self._calculate_performance_metrics()
        
        # 生成外层折汇总信息
        self._generate_outer_fold_summary()
        
        # 生成最终结果
        final_results = self._generate_final_results()
        
        self.logger.info("嵌套交叉验证评估完成")
        return final_results
    
    def _create_outer_cv(self, cohort_info: Optional[np.ndarray]):
        """创建外层CV分割器。"""
        if self.config.strategy == CVStrategy.LOCO and cohort_info is not None:
            return LeaveOneGroupOut()
        elif self.config.n_repeats > 1:
            # 重复CV
            return RepeatedStratifiedKFold(
                n_splits=self.config.outer_folds,
                n_repeats=self.config.n_repeats,
                random_state=self.config.random_state
            )
        else:
            return StratifiedKFold(
                n_splits=self.config.outer_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
    
    def _execute_inner_cv_optimization(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray, 
        models: Dict[str, BaseModel],
        outer_fold_idx: int
    ) -> Dict[str, Any]:
        """
        执行内层CV优化：特征选择 + 超参数调优。
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            models: 模型字典
            outer_fold_idx: 外层折索引
            
        Returns:
            内层CV优化结果
        """
        self.logger.info(f"  执行内层CV优化 (折 {outer_fold_idx + 1})...")
        
        # 创建内层CV分割器
        inner_cv = StratifiedKFold(
            n_splits=self.config.inner_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        inner_results = {
            'selected_features': {},
            'best_hyperparameters': {},
            'inner_fold_scores': {},
            'feature_importance_scores': {},
            'inner_fold_results': {}
        }

        # 统一内层搜索方法：优先从配置获取；默认 grid
        search_method = 'grid'
        try:
            # 首先尝试从实验配置获取
            if self.experiment_config and hasattr(self.experiment_config, 'search_method'):
                search_method = self.experiment_config.search_method
            # 然后尝试从特征选择配置获取
            elif hasattr(self, 'feature_selection_config') and getattr(self.feature_selection_config, 'search_method', None):
                search_method = getattr(self.feature_selection_config, 'search_method')
        except Exception:
            pass
        
        self.logger.info(f"  内层CV使用搜索方法: {search_method}")
        
        # 对每个模型执行内层CV
        for model_name, base_model in models.items():
            self.logger.info(f"    优化模型: {model_name}")
            
            # 存储内层折结果
            inner_fold_features = []
            inner_fold_scores = []
            inner_fold_hyperparams = []
            inner_fold_importance = []
            
            # 内层CV循环
            for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
                inner_cv.split(X_train, y_train)
            ):
                X_inner_train = X_train.iloc[inner_train_idx]
                X_inner_val = X_train.iloc[inner_val_idx]
                y_inner_train = y_train[inner_train_idx]
                y_inner_val = y_train[inner_val_idx]
                
                # 联合优化：特征选择 + 超参数调优
                selected_features, best_hyperparams = self._joint_feature_hyperparameter_optimization(
                    X_inner_train, y_inner_train, model_name, search_method
                )
                self.logger.info(f"      内层折 {inner_fold_idx + 1}: 从 {X_inner_train.shape[1]} 个特征中选择了 {len(selected_features)} 个特征")
                
                # 3. 训练模型并评估
                model_score, feature_importance = self._train_and_evaluate_model(
                    X_inner_train, X_inner_val, y_inner_train, y_inner_val,
                    model_name, selected_features, best_hyperparams
                )
                
                # 存储结果
                inner_fold_features.append(selected_features)
                inner_fold_scores.append(model_score)
                inner_fold_hyperparams.append(best_hyperparams)
                inner_fold_importance.append(feature_importance)
            
            # 计算内层CV的共识结果
            consensus_features = self._calculate_consensus_features_for_model(
                inner_fold_features, X_train.columns
            )
            
            best_hyperparams = self._select_best_hyperparameters(
                inner_fold_hyperparams, inner_fold_scores
            )
            
            # 存储模型结果
            inner_results['selected_features'][model_name] = consensus_features
            inner_results['best_hyperparameters'][model_name] = best_hyperparams
            inner_results['inner_fold_scores'][model_name] = inner_fold_scores
            inner_results['feature_importance_scores'][model_name] = inner_fold_importance
            inner_results['inner_fold_results'][model_name] = inner_fold_features
            
            self.logger.info(f"    {model_name}: 选择 {len(consensus_features)} 个特征")
        
        return inner_results

    def _map_method(self, method: str) -> str:
        mapping = {
            'grid': 'grid',
            'random': 'random',
            'bayes': 'bayes'
        }
        return mapping.get(method, 'grid')
    
    def _select_features(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        model_name: str
    ) -> List[str]:
        """选择特征，提供健壮的回退机制"""
        # 使用模型特定的特征选择策略
        try:
            selected_features = self.model_specific_selector.select_features(
                X, y, model_name
            )
            if not selected_features:
                raise ValueError("模型特定特征选择返回空结果")
            self.logger.debug(f"模型特定特征选择成功: {len(selected_features)} 个特征")
            return selected_features
        except Exception as e:
            self.logger.warning(f"模型特定特征选择失败: {e}，使用默认策略")
            
            # 改进的回退策略
            try:
                if model_name.lower() in ['lasso', 'elasticnet']:
                    # 对于L1正则化模型，返回所有特征让模型自己选择
                    self.logger.info("L1正则化模型：返回所有特征让模型自己选择")
                    return X.columns.tolist()
                else:
                    # 对于其他模型，使用SelectKBest
                    k = min(50, X.shape[1], max(10, X.shape[1] // 10))
                    self.logger.info(f"使用SelectKBest选择 {k} 个特征")
                    selector = self.feature_selector_factory.create_selector("selectkbest", k=k)
                    X_selected = selector.fit_transform(X, y)
                    selected_features = X_selected.columns.tolist()
                    if not selected_features:
                        raise ValueError("SelectKBest返回空结果")
                    return selected_features
            except Exception as fallback_error:
                self.logger.error(f"回退特征选择也失败: {fallback_error}")
                # 最后的回退：返回所有特征
                self.logger.warning("使用最后的回退策略：返回所有特征")
                return X.columns.tolist()
    
    # 旧模型特定调参方法已废弃；统一使用通用调参器
    # 保留空实现以避免外部调用报错
    def _tune_hyperparameters(self, *args, **kwargs) -> Dict[str, Any]:
        return {}
    
    def _train_and_evaluate_model(
        self, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame,
        y_train: np.ndarray, 
        y_val: np.ndarray,
        model_name: str,
        selected_features: List[str],
        hyperparams: Dict[str, Any]
    ) -> Tuple[float, Optional[np.ndarray]]:
        """训练模型并评估。"""
        # 选择特征
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
        
        # 创建和训练模型
        model = self._create_model(model_name, hyperparams)
        model.fit(X_train_selected, y_train)
        
        # 预测和评估
        y_pred_proba = model.predict_proba(X_val_selected)
        if y_pred_proba.shape[1] == 2:
            score = roc_auc_score(y_val, y_pred_proba[:, 1])
        else:
            score = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        
        # 获取特征重要性
        feature_importance = model.get_feature_importance()
        
        return score, feature_importance
    
    def _create_model(self, model_name: str, hyperparams: Dict[str, Any]) -> BaseModel:
        """创建模型实例。"""
        from ..models import (LassoClassifier, RandomForestClassifier, NeuralNetworkClassifier, 
                            ElasticNetClassifier, LogisticRegressionClassifier, CatBoostClassifier,
                            SVMClassifier, XGBoostClassifier, KNNClassifier, GaussianNBClassifier)
        
        if model_name.lower() == 'lasso':
            return LassoClassifier(**hyperparams)
        elif model_name.lower() == 'elasticnet':
            return ElasticNetClassifier(**hyperparams)
        elif model_name.lower() == 'logistic':
            return LogisticRegressionClassifier(**hyperparams)
        elif model_name.lower() == 'randomforest':
            return RandomForestClassifier(**hyperparams)
        elif model_name.lower() == 'catboost':
            return CatBoostClassifier(**hyperparams)
        elif model_name.lower() == 'neuralnetwork':
            return NeuralNetworkClassifier(**hyperparams)
        elif model_name.lower() == 'svm':
            return SVMClassifier(**hyperparams)
        elif model_name.lower() == 'xgboost':
            return XGBoostClassifier(**hyperparams)
        elif model_name.lower() == 'knn':
            return KNNClassifier(**hyperparams)
        elif model_name.lower() == 'gaussiannb':
            return GaussianNBClassifier(**hyperparams)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _calculate_consensus_features_for_model(
        self, 
        inner_fold_features: List[List[str]], 
        all_features: pd.Index
    ) -> List[str]:
        """计算内层CV的共识特征。"""
        # 统计每个特征被选择的频率
        feature_counts = Counter()
        for fold_features in inner_fold_features:
            for feature in fold_features:
                feature_counts[feature] += 1
        
        # 选择被选择频率 >= 50% 的特征作为共识特征
        consensus_threshold = len(inner_fold_features) * 0.5
        consensus_features = [
            feature for feature, count in feature_counts.items()
            if count >= consensus_threshold
        ]
        
        return consensus_features
    
    def _joint_feature_hyperparameter_optimization(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        model_name: str,
        search_method: str
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        联合优化特征选择和超参数调优。
        
        策略：
        1. 先进行特征选择，得到候选特征集
        2. 对候选特征集进行超参数调优
        3. 如果模型支持特征重要性，基于重要性进一步筛选特征
        4. 可选：多轮迭代优化（特征选择 -> 超参数调优 -> 特征筛选）
        """
        try:
            from .model_specific_hyperparameter_tuner import ModelSpecificHyperparameterTuner
            from .hyperparameter_tuner import HyperparameterTuner
            ms_tuner = ModelSpecificHyperparameterTuner()
            
            # 计算类别平衡信息
            class_balance = self._calculate_class_balance(y)
            
            # 1. 先进行特征选择
            selected_features = self._select_features(X, y, model_name)
            
            if not selected_features:
                self.logger.warning(f"特征选择返回空结果，使用所有特征")
                selected_features = X.columns.tolist()
            
            # 2. 使用选出的特征进行超参数调优
            param_grid = ms_tuner.get_adaptive_param_grid(
                model_name, 
                X[selected_features].shape, 
                len(y), 
                len(selected_features),
                class_balance
            )
            
            # 记录自适应参数信息（仅记录一次）
            if model_name not in self.adaptive_param_info_:
                # 获取实际的类别标签（传递标签映射信息）
                label_mapping = getattr(self, 'label_mapping_', None)
                actual_labels = self._get_actual_class_labels(y, label_mapping=label_mapping)
                
                self.adaptive_param_info_[model_name] = {
                    'data_shape': X[selected_features].shape,
                    'n_samples': len(y),
                    'n_features': len(selected_features),
                    'class_balance': class_balance,
                    'param_grid': param_grid,
                    'search_method': search_method,
                    'class_labels': actual_labels,
                    'optimization_method': 'joint_feature_hyperparameter',  # 标记使用联合优化
                    'feature_selection_method': 'model_specific',  # 特征选择方法
                    'hyperparameter_tuning_method': search_method,  # 超参数调优方法
                    'importance_based_refinement': True,  # 是否使用重要性筛选
                    'inner_cv_folds': self.config.inner_folds,  # 内层CV折数
                    'cv_random_state': self.config.random_state  # CV随机种子
                }
            
            # 3. 超参数调优
            tuner = HyperparameterTuner(method=self._map_method(search_method), n_jobs=self.config.n_jobs)
            base_model = self._create_model(model_name, {})
            tuned_model = tuner.tune(
                base_model,
                X[selected_features],
                y,
                param_grid,
                cv=self.config.inner_folds
            )
            best_hyperparams = getattr(tuner, 'best_params_', {})
            
            # 4. 基于特征重要性进一步筛选特征（如果支持）
            final_features = self._refine_features_by_importance(
                tuned_model, selected_features, X, y, model_name
            )
            
            return final_features, best_hyperparams
            
        except Exception as e:
            self.logger.warning(f"联合优化失败，回退到分离式优化: {e}")
            # 回退到原来的分离式方法
            selected_features = self._select_features(X, y, model_name)
            best_hyperparams = self._tune_hyperparameters(X, y, model_name, selected_features)
            return selected_features, best_hyperparams

    def _refine_features_by_importance(
        self, 
        model, 
        selected_features: List[str], 
        X: pd.DataFrame, 
        y: np.ndarray, 
        model_name: str
    ) -> List[str]:
        """
        基于特征重要性进一步筛选特征。
        """
        try:
            if not hasattr(model, 'get_feature_importance'):
                return selected_features
                
            feature_importance = model.get_feature_importance()
            if feature_importance is None or len(feature_importance) != len(selected_features):
                return selected_features
            
            # 基于重要性筛选特征
            # 策略1：保留重要性前80%的特征
            importance_threshold = np.percentile(feature_importance, 20)  # 保留前80%
            important_mask = feature_importance >= importance_threshold
            
            # 策略2：确保至少保留一定数量的特征
            min_features = max(10, len(selected_features) // 4)  # 至少保留1/4的特征
            if np.sum(important_mask) < min_features:
                # 如果重要特征太少，选择重要性最高的min_features个
                top_indices = np.argsort(feature_importance)[-min_features:]
                important_mask = np.zeros(len(selected_features), dtype=bool)
                important_mask[top_indices] = True
            
            if np.sum(important_mask) > 0:
                refined_features = [f for f, keep in zip(selected_features, important_mask) if keep]
                self.logger.debug(f"基于重要性筛选: {len(selected_features)} -> {len(refined_features)} 个特征")
                return refined_features
            else:
                return selected_features
                
        except Exception as e:
            self.logger.debug(f"基于重要性筛选失败: {e}")
            return selected_features

    def _select_best_hyperparameters(
        self, 
        hyperparams_list: List[Dict[str, Any]], 
        scores: List[float]
    ) -> Dict[str, Any]:
        """选择最佳超参数。"""
        # 选择得分最高的超参数组合
        best_idx = np.argmax(scores)
        return hyperparams_list[best_idx]
    
    def _evaluate_outer_fold(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: np.ndarray, 
        y_test: np.ndarray,
        models: Dict[str, BaseModel],
        inner_results: Dict[str, Any],
        outer_fold_idx: int,
        repeat_idx: int = 0,
        train_sample_ids: Optional[List[str]] = None,
        test_sample_ids: Optional[List[str]] = None,
        cohort_info: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """评估外层折。"""
        self.logger.info(f"  评估外层折 {outer_fold_idx + 1}...")
        
        # 获取测试队列信息（仅LOCO模式）
        test_cohort_name = None
        if self.config.strategy == CVStrategy.LOCO and cohort_info is not None and test_idx is not None:
            # 直接使用原始数据的测试索引获取队列信息
            test_cohort_values = cohort_info[test_idx]
            
            # 获取唯一的测试队列名称
            unique_test_cohorts = list(set(test_cohort_values))
            if len(unique_test_cohorts) == 1:
                test_cohort_name = str(unique_test_cohorts[0])
                self.logger.info(f"  LOCO测试队列: {test_cohort_name}")
            else:
                self.logger.warning(f"  测试集包含多个队列: {unique_test_cohorts}")
                test_cohort_name = f"Mixed_{len(unique_test_cohorts)}_cohorts"
        
        fold_results = {
            'fold_idx': outer_fold_idx,
            'repeat_idx': repeat_idx,
            'test_samples': len(y_test),
            'test_sample_ids': test_sample_ids if test_sample_ids is not None else X_test.index.tolist(),
            'train_sample_ids': train_sample_ids if train_sample_ids is not None else X_train.index.tolist(),
            'y_test': y_test.tolist(),
            'y_train': y_train.tolist(),
            'test_cohort_name': test_cohort_name,  # 添加测试队列信息
            'model_results': {}
        }
        
        # 对每个模型评估
        for model_name, base_model in models.items():
            # 获取内层CV的共识特征和最佳超参数
            consensus_features = inner_results['selected_features'][model_name]
            best_hyperparams = inner_results['best_hyperparameters'][model_name]
            
            # 选择特征
            X_train_selected = X_train[consensus_features]
            X_test_selected = X_test[consensus_features]
            
            # 训练最终模型
            final_model = self._create_model(model_name, best_hyperparams)
            final_model.fit(X_train_selected, y_train)
            
            # 预测和评估
            y_pred = final_model.predict(X_test_selected)
            y_pred_proba = final_model.predict_proba(X_test_selected)
            
            # 计算指标
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            fold_results['model_results'][model_name] = {
                'selected_features': consensus_features,
                'hyperparameters': best_hyperparams,
                'metrics': metrics,
                'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                'probabilities': y_pred_proba.tolist() if hasattr(y_pred_proba, 'tolist') else list(y_pred_proba),
                'inner_selected_features': inner_results.get('inner_fold_results', {}).get(model_name, [])
            }
        
        return fold_results
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """计算评估指标。"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary')
        }
        
        # 计算AUC
        if y_pred_proba.shape[1] == 2:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        
        return metrics
    
    def _calculate_consensus_features(self):
        """计算所有外层折的共识特征，支持重复CV和LOCO。"""
        self.logger.info("计算共识特征...")
        
        # 使用共识特征选择器
        from .consensus_feature_selector import create_consensus_selector
        from .hyperparameter_analyzer import HyperparameterAnalyzer
        
        # 获取用户指定的阈值（从特征选择配置中获取）
        threshold = 0.5  # 默认阈值
        if self.feature_selection_config:
            threshold = self.feature_selection_config.threshold
        
        # 三层筛选的阈值策略：内层CV -> 外层CV -> 重复筛选
        n_folds = len(self.outer_fold_results_)
        
        # 使用用户指定的阈值
        # 注意：如果有联合优化和重要性筛选，用户可以考虑设置更宽松的阈值
        # 让更多特征进入超参数调优阶段，然后通过重要性筛选进一步优化
        adjusted_threshold = threshold
        
        if self.config.strategy == CVStrategy.LOCO:
            self.logger.info(f"LOCO两层筛选策略: 阈值={threshold:.3f}, 队列数={n_folds}")
            self.logger.info("  筛选流程: 内层CV筛选 -> 队列筛选")
        elif self.config.n_repeats > 1:
            self.logger.info(f"重复CV三层筛选策略: 阈值={threshold:.3f}, 总分割={n_folds}, 重复次数={self.config.n_repeats}")
            self.logger.info("  筛选流程: 内层CV筛选 -> 外层CV筛选 -> 重复筛选")
        else:
            self.logger.info(f"普通CV三层筛选策略: 阈值={threshold:.3f}, 折数={n_folds}")
            self.logger.info("  筛选流程: 内层CV筛选 -> 外层CV筛选 -> 重复筛选")
        
        # 创建选择器（使用调整后的阈值）
        selector = create_consensus_selector(
            strategy="majority_voting",
            threshold=adjusted_threshold,
            min_features=10,
            max_features=1000
        )
        
        # 选择共识特征
        self.consensus_features_ = selector.select_consensus_features(self.outer_fold_results_)
        
        # 记录选择统计信息
        self.selection_stats_ = selector.get_selection_stats()
        
        # 根据CV策略显示不同的日志信息
        if self.config.strategy == CVStrategy.LOCO:
            for model_name, features in self.consensus_features_.items():
                self.logger.info(f"{model_name}: {len(features)} 个共识特征 (LOCO: {len(self.outer_fold_results_)}个队列)")
        elif self.config.n_repeats > 1:
            for model_name, features in self.consensus_features_.items():
                self.logger.info(f"{model_name}: {len(features)} 个共识特征 (重复CV: {self.config.n_repeats}次)")
        else:
            for model_name, features in self.consensus_features_.items():
                self.logger.info(f"{model_name}: {len(features)} 个共识特征 (普通CV: {self.config.outer_folds}折)")
    
    def _calculate_performance_metrics(self):
        """计算性能指标，支持重复CV和LOCO。"""
        self.logger.info("计算性能指标...")
        
        for model_name in self.outer_fold_results_[0]['model_results'].keys():
            # 收集所有外层折的指标
            metrics_by_fold = []
            for fold_result in self.outer_fold_results_:
                metrics = fold_result['model_results'][model_name]['metrics']
                metrics_by_fold.append(metrics)
            
            # 计算平均指标和标准差
            model_metrics = {}
            for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                values = [m[metric_name] for m in metrics_by_fold]
                model_metrics[f'{metric_name}_mean'] = np.mean(values)
                model_metrics[f'{metric_name}_std'] = np.std(values)
                model_metrics[f'{metric_name}_values'] = values
            
            # 添加CV策略相关信息
            if self.config.strategy == CVStrategy.LOCO:
                model_metrics['cv_strategy'] = 'LOCO'
                model_metrics['n_cohorts'] = len(metrics_by_fold)
                self.logger.info(f"{model_name} LOCO性能: {len(metrics_by_fold)}个队列")
            elif self.config.n_repeats > 1:
                model_metrics['cv_strategy'] = 'RepeatedKFold'
                model_metrics['n_repeats'] = self.config.n_repeats
                model_metrics['n_folds_per_repeat'] = self.config.outer_folds
                model_metrics['total_folds'] = len(metrics_by_fold)
                self.logger.info(f"{model_name} 重复CV性能: {len(metrics_by_fold)}个分割, {self.config.n_repeats}次重复")
            else:
                model_metrics['cv_strategy'] = 'StratifiedKFold'
                model_metrics['n_folds'] = len(metrics_by_fold)
                self.logger.info(f"{model_name} 普通CV性能: {len(metrics_by_fold)}个分割")
            
            self.performance_metrics_[model_name] = model_metrics
    
    def _generate_outer_fold_summary(self):
        """生成外层折汇总信息。"""
        self.logger.info("生成外层折汇总信息...")
        
        # 初始化汇总信息
        self.outer_fold_summary_ = {
            'cv_strategy_info': {},
            'fold_statistics': {},
            'model_summary': {}
        }
        
        # CV策略信息
        if self.config.strategy == CVStrategy.LOCO:
            self.outer_fold_summary_['cv_strategy_info'] = {
                'strategy': 'LOCO',
                'n_cohorts': len(self.outer_fold_results_),
                'description': 'Leave-One-Cohort-Out Cross-Validation'
            }
        elif self.config.n_repeats > 1:
            self.outer_fold_summary_['cv_strategy_info'] = {
                'strategy': 'RepeatedStratifiedKFold',
                'n_repeats': self.config.n_repeats,
                'n_folds_per_repeat': self.config.outer_folds,
                'total_folds': len(self.outer_fold_results_),
                'description': f'Repeated Stratified K-Fold ({self.config.n_repeats} repeats, {self.config.outer_folds} folds each)'
            }
        else:
            self.outer_fold_summary_['cv_strategy_info'] = {
                'strategy': 'StratifiedKFold',
                'n_folds': len(self.outer_fold_results_),
                'description': 'Stratified K-Fold Cross-Validation'
            }
        
        # 折统计信息
        test_sample_counts = [fold['test_samples'] for fold in self.outer_fold_results_]
        
        if self.config.strategy == CVStrategy.LOCO:
            # LOCO模式：显示队列统计信息
            self.outer_fold_summary_['fold_statistics'] = {
                'total_cohorts': len(self.outer_fold_results_),
                'test_samples_per_cohort': {
                    'mean': float(np.mean(test_sample_counts)),
                    'std': float(np.std(test_sample_counts)),
                    'min': int(np.min(test_sample_counts)),
                    'max': int(np.max(test_sample_counts)),
                    'values': test_sample_counts
                }
            }
        else:
            # 普通CV和重复CV模式：显示折统计信息
            self.outer_fold_summary_['fold_statistics'] = {
                'total_folds': len(self.outer_fold_results_),
                'test_samples_per_fold': {
                    'mean': float(np.mean(test_sample_counts)),
                    'std': float(np.std(test_sample_counts)),
                    'min': int(np.min(test_sample_counts)),
                    'max': int(np.max(test_sample_counts)),
                    'values': test_sample_counts
                }
            }
        
        # 模型汇总信息
        for model_name in self.consensus_features_.keys():
            if model_name in self.performance_metrics_:
                metrics = self.performance_metrics_[model_name]
                self.outer_fold_summary_['model_summary'][model_name] = {
                    'consensus_features_count': len(self.consensus_features_[model_name]),
                    'performance_summary': {
                        'accuracy': {
                            'mean': metrics.get('accuracy_mean', 0),
                            'std': metrics.get('accuracy_std', 0)
                        },
                        'auc': {
                            'mean': metrics.get('auc_mean', 0),
                            'std': metrics.get('auc_std', 0)
                        },
                        'f1': {
                            'mean': metrics.get('f1_mean', 0),
                            'std': metrics.get('f1_std', 0)
                        }
                    }
                }
        
        self.logger.info(f"外层折汇总信息生成完成: {len(self.outer_fold_results_)}个折")
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """生成最终结果。"""
        return {
            'nested_cv_results': {
                'outer_fold_results': self.outer_fold_results_,
                'consensus_features': self.consensus_features_,
                'performance_metrics': self.performance_metrics_
            },
            'summary': {
                'n_outer_folds': len(self.outer_fold_results_),
                'n_models': len(self.consensus_features_),
                'consensus_features_count': {
                    model: len(features) 
                    for model, features in self.consensus_features_.items()
                }
            }
        }
    
    def get_results(self) -> Dict[str, Any]:
        """获取评估结果。"""
        return self._generate_final_results()
    
    def _calculate_class_balance(self, y: np.ndarray) -> float:
        """计算类别平衡比例。"""
        unique_labels, counts = np.unique(y, return_counts=True)
        if len(unique_labels) != 2:
            return 0.5  # 非二分类，返回中性值
        
        # 返回正类（第二个类别）的比例
        positive_count = counts[1] if len(counts) > 1 else counts[0]
        total_count = len(y)
        return positive_count / total_count
    

    def _get_actual_class_labels(self, y: np.ndarray, metadata: Optional[pd.DataFrame] = None, group_col: str = "Group", label_mapping: Optional[Dict[str, int]] = None) -> Dict[str, str]:
        """获取实际的类别标签名称。"""
        unique_labels, counts = np.unique(y, return_counts=True)
        
        if len(unique_labels) != 2:
            return {"positive_class": "Class_1", "negative_class": "Class_0"}
        
        # 如果有明确的标签映射，使用它
        if label_mapping is not None:
            # 反转映射：从 {label: 0/1} 到 {0/1: label}
            reverse_mapping = {v: k for k, v in label_mapping.items()}
            return {
                "positive_class": str(reverse_mapping.get(1, f"Class_{unique_labels[1]}")),
                "negative_class": str(reverse_mapping.get(0, f"Class_{unique_labels[0]}"))
            }
        
        # 如果有metadata，尝试从Group列获取实际标签
        if metadata is not None and group_col in metadata.columns:
            try:
                # 获取Group列的唯一值
                group_values = metadata[group_col].unique()
                if len(group_values) == 2:
                    # 按照y的排序顺序映射
                    sorted_groups = sorted(group_values)
                    return {
                        "positive_class": str(sorted_groups[1]),  # 正类（标签1）
                        "negative_class": str(sorted_groups[0])   # 负类（标签0）
                    }
            except Exception:
                pass
        
        # 回退到默认标签
        return {
            "positive_class": f"Class_{unique_labels[1]}",
            "negative_class": f"Class_{unique_labels[0]}"
        }
