"""
嵌套CV分类器 for metaClassifier v1.0.

这个模块实现了第一部分：模型评估（嵌套CV阶段）
- 目的：无偏估计模型泛化性能 + 筛选稳定特征
- 做法：内层CV（特征选择+超参数调优）+ 外层CV（性能评估）
- 输出：外层平均性能指标 + 共识特征集
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from .base import BaseClassifier, ExperimentConfig, TaskType, CVStrategy, CVConfig
from .nested_cv_evaluator import NestedCVEvaluator
from .standard_output_manager import StandardOutputManager
from ..data.loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from ..data.validator import DataValidator
from ..preprocessing.variance_filter import AdaptiveVarianceFilter, AdaptiveFilterConfig
from ..evaluation.visualizer import ResultsVisualizer
from ..utils.logger import get_logger


class NestedCVClassifier(BaseClassifier):
    """
    嵌套CV分类器 - 第一部分：模型评估。
    
    实现完整的嵌套CV逻辑：
    1. 外层CV：无偏估计模型泛化性能
    2. 内层CV：特征选择 + 超参数调优
    3. 输出：外层平均性能指标 + 共识特征集
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化嵌套CV分类器。
        
        Args:
            config: 实验配置
        """
        super().__init__(config)
        self.logger = get_logger("NestedCVClassifier")
        
        # 初始化组件
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()
        self.data_validator = DataValidator()
        self.adaptive_filter = AdaptiveVarianceFilter(config.adaptive_filter)
        self.visualizer = ResultsVisualizer()
        self.output_manager = StandardOutputManager()
        
        # 创建嵌套CV评估器
        self.nested_cv_evaluator = NestedCVEvaluator(config.cv, config.feature_selection, config)
        
        # 结果存储
        self.evaluation_results_ = None
        self.consensus_features_ = None
        self.performance_metrics_ = None
        
        self.logger.info(f"NestedCVClassifier initialized for {config.model.name}")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        cohort_info: Optional[np.ndarray] = None,
        original_features: Optional[List[str]] = None,
        constant_removed_features: Optional[List[str]] = None
    ) -> 'NestedCVClassifier':
        """
        拟合嵌套CV分类器。
        
        注意：这个类主要用于评估，不进行传统的拟合。
        实际的模型训练在evaluate方法中进行。
        """
        self.logger.info("NestedCVClassifier: 开始数据预处理...")
        
        # 验证数据
        self.data_validator.validate(X, y, cohort_info)
        
        # 预处理数据
        X_processed = self.data_preprocessor.fit_transform(X, y)
        
        # 应用自适应方差过滤
        if self.config.adaptive_filter.enabled:
            X_processed, filter_info = self.adaptive_filter.filter_features(X_processed, y)
            self.logger.info(f"应用自适应方差过滤: {filter_info}")
        
        # 存储处理后的数据
        self.X_processed_ = X_processed
        self.y_processed_ = y
        self.cohort_info_ = cohort_info
        # 记录原始与过滤信息用于后续特征追踪
        try:
            # 使用传入的原始特征名称，如果没有则使用当前X的列名
            self.original_feature_names_ = original_features if original_features is not None else list(X.columns)
        except Exception:
            self.original_feature_names_ = None
        try:
            # 使用传入的常量特征移除信息，如果没有则使用数据预处理器的信息
            if constant_removed_features is not None:
                self.constant_removed_ = set(constant_removed_features)
            else:
                self.constant_removed_ = set(self.data_preprocessor.constant_features_ or [])
        except Exception:
            self.constant_removed_ = set()
        try:
            self.variance_removed_ = set(self.adaptive_filter.removed_features_ or []) if self.config.adaptive_filter.enabled else set()
        except Exception:
            self.variance_removed_ = set()
        
        self.logger.info("数据预处理完成")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测方法。
        
        注意：NestedCVClassifier主要用于评估，不提供预测功能。
        预测功能在最终模型确定阶段提供。
        """
        raise NotImplementedError(
            "NestedCVClassifier主要用于模型评估，不提供预测功能。"
            "请使用最终模型进行预测。"
        )
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率方法。
        
        注意：NestedCVClassifier主要用于评估，不提供预测功能。
        预测功能在最终模型确定阶段提供。
        """
        raise NotImplementedError(
            "NestedCVClassifier主要用于模型评估，不提供预测功能。"
            "请使用最终模型进行预测。"
        )
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        cohort_info: Optional[np.ndarray] = None,
        original_features: Optional[List[str]] = None,
        constant_removed_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行嵌套CV评估。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            cohort_info: 队列信息（用于LOCO）
            
        Returns:
            嵌套CV评估结果
        """
        self.logger.info("开始嵌套CV评估...")
        
        # 先进行数据预处理
        self.fit(X, y, cohort_info, original_features, constant_removed_features)
        
        # 创建模型
        models = self._create_models()
        
        # 执行嵌套CV评估
        self.evaluation_results_ = self.nested_cv_evaluator.evaluate(
            self.X_processed_, self.y_processed_, models, self.cohort_info_
        )
        
        # 提取结果
        self.consensus_features_ = self.evaluation_results_['nested_cv_results']['consensus_features']
        self.performance_metrics_ = self.evaluation_results_['nested_cv_results']['performance_metrics']
        
        # 传递外层CV特有信息
        if hasattr(self.nested_cv_evaluator, 'selection_stats_'):
            self.selection_stats_ = self.nested_cv_evaluator.selection_stats_
        if hasattr(self.nested_cv_evaluator, 'outer_fold_summary_'):
            self.outer_fold_summary_ = self.nested_cv_evaluator.outer_fold_summary_
        if hasattr(self.nested_cv_evaluator, 'adaptive_param_info_'):
            self.adaptive_param_info_ = self.nested_cv_evaluator.adaptive_param_info_
        if hasattr(self.nested_cv_evaluator, 'class_balance_info_'):
            self.class_balance_info_ = self.nested_cv_evaluator.class_balance_info_
        if hasattr(self.nested_cv_evaluator, 'label_mapping_'):
            self.label_mapping_ = self.nested_cv_evaluator.label_mapping_
        
        # 生成报告和可视化
        self._generate_evaluation_outputs()
        
        self.logger.info("嵌套CV评估完成")
        return self.evaluation_results_
    
    def _create_models(self) -> Dict[str, Any]:
        """创建要评估的模型。"""
        from ..models import (LassoClassifier, RandomForestClassifier, NeuralNetworkClassifier, 
                            ElasticNetClassifier, LogisticRegressionClassifier, CatBoostClassifier,
                            SVMClassifier, XGBoostClassifier, KNNClassifier, GaussianNBClassifier)
        
        models = {}
        
        # 根据配置创建模型
        if self.config.model.name.lower() == 'lasso':
            models['Lasso'] = LassoClassifier(**self.config.model.hyperparameters)
        elif self.config.model.name.lower() == 'elasticnet':
            models['ElasticNet'] = ElasticNetClassifier(**self.config.model.hyperparameters)
        elif self.config.model.name.lower() == 'logistic':
            models['Logistic'] = LogisticRegressionClassifier(**self.config.model.hyperparameters)
        elif self.config.model.name.lower() == 'randomforest':
            models['RandomForest'] = RandomForestClassifier(**self.config.model.hyperparameters)
        elif self.config.model.name.lower() == 'catboost':
            models['CatBoost'] = CatBoostClassifier(**self.config.model.hyperparameters)
        elif self.config.model.name.lower() == 'neuralnetwork':
            models['NeuralNetwork'] = NeuralNetworkClassifier(**self.config.model.hyperparameters)
        elif self.config.model.name.lower() == 'svm':
            models['SVM'] = SVMClassifier(**self.config.model.hyperparameters)
        elif self.config.model.name.lower() == 'xgboost':
            models['XGBoost'] = XGBoostClassifier(**self.config.model.hyperparameters)
        elif self.config.model.name.lower() == 'knn':
            models['KNN'] = KNNClassifier(**self.config.model.hyperparameters)
        elif self.config.model.name.lower() == 'gaussiannb':
            models['GaussianNB'] = GaussianNBClassifier(**self.config.model.hyperparameters)
        else:
            # 默认创建所有模型
            models['Lasso'] = LassoClassifier()
            models['RandomForest'] = RandomForestClassifier()
            models['NeuralNetwork'] = NeuralNetworkClassifier()
        
        return models
    
    def _generate_evaluation_outputs(self):
        """生成评估输出（使用标准输出管理器）。"""
        if not self.evaluation_results_:
            return
        
        # 创建输出目录
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用标准输出管理器保存结果
        self._save_standard_outputs(output_dir)
    
    def _save_standard_outputs(self, output_dir: Path):
        """使用标准输出管理器保存结果。"""
        self.logger.info("保存标准输出格式...")
        
        # 获取模型名称
        model_name = self.config.model.name.lower()
        
        # 1. 保存性能指标
        if self.evaluation_results_:
            self.output_manager.save_performance_metrics(
                output_dir, self.evaluation_results_, model_name, self.visualizer
            )
        
        # 2. 保存特征分析
        if self.consensus_features_:
            self.output_manager.save_feature_analysis(
                output_dir, self.consensus_features_, 
                feature_importance=None, model_name=model_name,
                nested_cv_results=self.evaluation_results_,
                all_features=self.original_feature_names_,
                constant_removed=self.constant_removed_,
                variance_removed=self.variance_removed_
            )
        
        # 3. 保存可重现性信息
        # 注意：嵌套CV阶段不保存最终的超参数，因为最终模型会在全量数据上重新训练超参数
        config_info = {
            'model_name': model_name,
            'cv_strategy': self.config.cv.strategy.value if hasattr(self.config.cv.strategy, 'value') else self.config.cv.strategy,
            'outer_folds': self.config.cv.outer_folds,
            'inner_folds': self.config.cv.inner_folds,
            'n_repeats': self.config.cv.n_repeats,
            'random_state': self.config.cv.random_state,
            'n_jobs': self.config.cv.n_jobs,
            'enable_adaptive_filtering': self.config.adaptive_filter.enabled,
            'adaptive_filter_config': self.config.adaptive_filter.__dict__ if hasattr(self.config.adaptive_filter, '__dict__') else {},
            'model_hyperparameters': {},  # 嵌套CV阶段不保存最终超参数，最终超参数在final_model_trainer中保存
            'model_task_type': self.config.model.task_type.value if hasattr(self.config.model, 'task_type') and hasattr(self.config.model.task_type, 'value') else (self.config.model.task_type if hasattr(self.config.model, 'task_type') else 'classification'),
            'feature_selection_config': self.config.feature_selection.__dict__ if hasattr(self.config, 'feature_selection') and self.config.feature_selection else {},
            'data_paths': self.config.data_paths if hasattr(self.config, 'data_paths') and self.config.data_paths else {},
            'preprocessing_config': self.config.model.preprocessing if hasattr(self.config.model, 'preprocessing') and self.config.model.preprocessing else {}
        }
        
        self.output_manager.save_reproducibility_info(
            output_dir, config_info, model_name
        )
    
    def get_consensus_features(self) -> Dict[str, List[str]]:
        """获取共识特征。"""
        return self.consensus_features_ or {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标。"""
        return self.performance_metrics_ or {}
    
    def get_evaluation_results(self) -> Optional[Dict[str, Any]]:
        """获取完整的评估结果。"""
        return self.evaluation_results_


def create_nested_cv_classifier(
    model_name: str = "Lasso",
    cv_strategy: CVStrategy = CVStrategy.REPEATED_KFOLD,
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_repeats: int = 1,
    enable_adaptive_filtering: bool = True,
    output_dir: str = "./nested_cv_results",
    **kwargs
) -> NestedCVClassifier:
    """
    创建嵌套CV分类器的工厂函数。
    
    Args:
        model_name: 模型名称
        cv_strategy: CV策略
        outer_folds: 外层CV折数
        inner_folds: 内层CV折数
        enable_adaptive_filtering: 是否启用自适应过滤
        output_dir: 输出目录
        **kwargs: 其他配置参数
        
    Returns:
        配置好的NestedCVClassifier实例
    """
    from .base import ModelConfig, CVConfig, AdaptiveFilterConfig, ExperimentConfig
    
    # 创建配置
    model_config = ModelConfig(
        name=model_name,
        task_type=TaskType.CLASSIFICATION,
        hyperparameters=kwargs.get('model_params', {})
    )
    
    cv_config = CVConfig(
        strategy=cv_strategy,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        n_repeats=n_repeats,
        random_state=kwargs.get('random_state', 42),
        n_jobs=kwargs.get('n_jobs', 1)
    )
    
    adaptive_filter_config = AdaptiveFilterConfig(
        enabled=enable_adaptive_filtering,
        min_q=kwargs.get('min_q', 0.05),
        max_q=kwargs.get('max_q', 0.2),
        r_mid=kwargs.get('r_mid', 5.0),
        steepness=kwargs.get('steepness', 0.5)
    )
    
    experiment_config = ExperimentConfig(
        model=model_config,
        cv=cv_config,
        adaptive_filter=adaptive_filter_config,
        data_paths=kwargs.get('data_paths', {}),
        output_dir=output_dir,
        verbose=kwargs.get('verbose', True),
        search_method=kwargs.get('search_method', 'grid')
    )
    
    return NestedCVClassifier(experiment_config)
