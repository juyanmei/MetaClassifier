"""
最终模型训练器 for metaClassifier v1.0.

这个模块实现了第二步：最终模型确定阶段
- 输入：第一部分得到的共识特征集 + 性能指标
- 过程：在全量数据上，使用共识特征集，只进行超参数调优，不进行特征选择
- 输出：形成最终的模型
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime

from .base import BaseClassifier, ExperimentConfig, TaskType, CVStrategy
from .hyperparameter_tuner import HyperparameterTuner
from ..data.loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from ..data.validator import DataValidator
from ..preprocessing.variance_filter import AdaptiveVarianceFilter, AdaptiveFilterConfig
from ..evaluation.metrics import MetricsCalculator
from ..evaluation.visualizer import ResultsVisualizer
from ..evaluation.reporter import ResultsReporter
from ..utils.logger import get_logger
from .standard_output_manager import StandardOutputManager


class FinalModelTrainer:
    """
    最终模型训练器 - 第二步：最终模型确定。
    
    基于第一部分的结果，在全量数据上训练最终模型：
    1. 使用共识特征集（不进行特征选择）
    2. 只进行超参数调优
    3. 形成最终的可用模型
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化最终模型训练器。
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.logger = get_logger("FinalModelTrainer")
        
        # 初始化组件
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()
        self.data_validator = DataValidator()
        self.adaptive_filter = AdaptiveVarianceFilter(config.adaptive_filter)
        self.hyperparameter_tuner = HyperparameterTuner()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = ResultsVisualizer()
        self.reporter = ResultsReporter()
        self.output_manager = StandardOutputManager()
        
        # 存储结果
        self.consensus_features_ = None
        self.performance_metrics_ = None
        self.final_models_ = {}
        self.training_results_ = None
        
        self.logger.info(f"FinalModelTrainer initialized for {config.model.name}")
    
    def load_nested_cv_results(self, results_path: Union[str, Path]) -> 'FinalModelTrainer':
        """
        加载第一部分（嵌套CV）的结果。
        
        Args:
            results_path: 嵌套CV结果文件路径
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Loading nested CV results from {results_path}")
        
        results_path = Path(results_path)
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        # 加载JSON结果
        import json
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        # 提取共识特征和性能指标
        nested_cv_results = results_data.get('nested_cv_results', {})
        self.consensus_features_ = nested_cv_results.get('consensus_features', {})
        self.performance_metrics_ = nested_cv_results.get('performance_metrics', {})
        
        self.logger.info(f"Loaded consensus features for {len(self.consensus_features_)} models")
        for model_name, features in self.consensus_features_.items():
            self.logger.info(f"  {model_name}: {len(features)} features")
        
        return self
    
    def set_consensus_features(self, consensus_features: Dict[str, List[str]]) -> 'FinalModelTrainer':
        """
        直接设置共识特征集。
        
        Args:
            consensus_features: 共识特征字典
            
        Returns:
            Self for method chaining
        """
        self.consensus_features_ = consensus_features
        self.logger.info(f"Set consensus features for {len(consensus_features)} models")
        return self
    
    def set_performance_metrics(self, performance_metrics: Dict[str, Any]) -> 'FinalModelTrainer':
        """
        直接设置性能指标。
        
        Args:
            performance_metrics: 性能指标字典
            
        Returns:
            Self for method chaining
        """
        self.performance_metrics_ = performance_metrics
        self.logger.info(f"Set performance metrics for {len(performance_metrics)} models")
        return self
    
    def train_final_models(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        cohort_info: Optional[np.ndarray] = None,
        final_cv_folds: int = 5,
        final_search_method: str = 'grid',
        cpu: int = 4
    ) -> Dict[str, Any]:
        """
        训练最终模型。
        
        Args:
            X: 特征矩阵
            y: 目标标签
            cohort_info: 队列信息（可选）
            
        Returns:
            最终模型训练结果
        """
        self.logger.info("开始训练最终模型...")
        
        if self.consensus_features_ is None:
            raise ValueError("Consensus features must be set before training final models")
        
        # 验证数据
        self.data_validator.validate(X, y, cohort_info)
        
        # 记录最终阶段使用的调参与资源参数（用于写入final_run.yaml）
        self.final_cv_folds_used = final_cv_folds
        self.final_search_method_used = final_search_method
        self.final_cpu_used = cpu

        # 预处理数据（禁用标准化/缩放，直接使用原始相对丰度）
        X_processed = X.copy()

        # 记录原始数据与样本ID，供后续在 save_final_models 中生成最终预测与HI
        try:
            self.data_loader.last_X_ = X.copy()
            self.data_loader.last_y_ = y.copy() if hasattr(y, 'copy') else y
            self.data_loader.last_sample_ids_ = getattr(X, 'index', None)
        except Exception:
            # 兜底：不抛错，稍后逻辑会检测并跳过
            self.data_loader.last_X_ = X
            self.data_loader.last_y_ = y
            self.data_loader.last_sample_ids_ = getattr(X, 'index', None)
        
        # 注意：最终模型训练时不重新应用自适应方差过滤
        # 因为共识特征是基于嵌套CV阶段的过滤结果确定的
        # 如果重新过滤，会导致特征不匹配
        
        # 训练每个模型的最终版本
        self.final_models_ = {}
        training_results = {}
        
        for model_name, consensus_features in self.consensus_features_.items():
            self.logger.info(f"训练最终模型: {model_name}")
            
            # 选择共识特征
            X_selected = X_processed[consensus_features]
            self.logger.info(f"  使用 {len(consensus_features)} 个共识特征")
            
            # 创建模型
            final_model = self._create_model(model_name)
            
            # 最终阶段：在全量数据上对共识特征做选参CV
            try:
                # 从模型特定模块获取参数空间
                from .model_specific_hyperparameter_tuner import ModelSpecificHyperparameterTuner
                ms_tuner = ModelSpecificHyperparameterTuner()
                
                # 根据搜索方法选择参数网格
                if final_search_method == 'bayes':
                    param_grid = ms_tuner.get_bayesian_param_grid(model_name)
                    self.logger.info(f"  使用贝叶斯优化参数网格: {param_grid}")
                else:
                    param_grid = ms_tuner.get_param_grid(model_name)
                    self.logger.info(f"  使用网格/随机搜索参数网格: {param_grid}")
                
                self.logger.info(f"  最终阶段使用搜索方法: {final_search_method}")
                tuner = HyperparameterTuner(method=self._map_method(final_search_method), n_jobs=cpu)
                tuned_model = tuner.tune(final_model, X_selected, y, param_grid, cv=final_cv_folds)
                final_model = tuned_model
                # 获取超参数调优结果
                best_params = getattr(tuner.tuner, 'best_params_', {}) if hasattr(tuner, 'tuner') else {}
                self.logger.info(f"  最终阶段选出的超参数: {best_params}")
            except Exception as e:
                self.logger.warning(f"  最终阶段选参失败，使用默认参数: {e}")
                # 如果调优失败，才需要手动训练
                final_model.fit(X_selected, y)
            # 在模型对象上挂载所用特征，便于推理阶段无需依赖外部文件
            try:
                setattr(final_model, 'selected_features_', list(consensus_features))
            except Exception:
                pass
            self.final_models_[model_name] = final_model
            
            # 评估最终模型
            final_metrics = self._evaluate_final_model(final_model, X_selected, y)
            training_results[model_name] = {
                'model': final_model,
                'consensus_features': consensus_features,
                'metrics': final_metrics,
                'hyperparameters': final_model.get_params() if hasattr(final_model, 'get_params') else {}
            }
            
            self.logger.info(f"  {model_name} 最终模型训练完成")
            self.logger.info(f"    准确率: {final_metrics['accuracy']:.4f}")
            self.logger.info(f"    AUC: {final_metrics['auc']:.4f}")
        
        self.training_results_ = training_results
        self.logger.info("所有最终模型训练完成")
        # 保存最终模型特征重要性到特征分析文件（追加列 final_importance）
        try:
            for model_name, model in self.final_models_.items():
                if hasattr(model, 'get_feature_importance'):
                    import numpy as np
                    import pandas as pd
                    importance = model.get_feature_importance()
                    if importance is not None and self.consensus_features_ and model_name in self.consensus_features_:
                        feat_names = self.consensus_features_[model_name]
                        
                        # 确保特征重要性长度与特征名称长度一致
                        if hasattr(importance, 'tolist'):
                            importance = importance.tolist()
                        
                        # 修复长度不匹配问题
                        if len(importance) != len(feat_names):
                            self.logger.warning(f"特征重要性长度({len(importance)})与特征名称长度({len(feat_names)})不匹配，进行修复")
                            if len(importance) > len(feat_names):
                                # 截取前N个元素
                                importance = importance[:len(feat_names)]
                            else:
                                # 用0填充
                                padding = [0.0] * (len(feat_names) - len(importance))
                                importance = importance + padding
                        
                        df = pd.DataFrame({'feature_name': feat_names, 'final_importance': importance})
                        # 读取已有的 consensus_features.csv 并合并
                        target_root = Path(self.config.output_dir) if self.config.output_dir else Path('.')
                        analysis_dir = target_root / '3_feature_analysis'
                        csv_path = analysis_dir / 'consensus_features.csv'
                        if csv_path.exists():
                            base_df = pd.read_csv(csv_path)
                            merged = base_df.merge(df, on='feature_name', how='left')
                            merged.to_csv(csv_path, index=False)
                            self.logger.info(f"成功更新特征重要性: {model_name}, 非零重要性数量: {sum(1 for x in importance if x != 0)}")
        except Exception as e:
            self.logger.warning(f"写入最终模型特征重要性失败: {e}")
        
        return training_results

    def _map_method(self, method: str) -> str:
        mapping = {
            'grid': 'grid',
            'random': 'random',
            'bayes': 'bayes'
        }
        return mapping.get(method, 'grid')
    
    def _create_model(self, model_name: str):
        """创建模型实例。"""
        from ..models import (LassoClassifier, RandomForestClassifier, NeuralNetworkClassifier, 
                            ElasticNetClassifier, LogisticRegressionClassifier, CatBoostClassifier,
                            SVMClassifier, XGBoostClassifier, KNNClassifier, GaussianNBClassifier)
        
        if model_name.lower() == 'lasso':
            return LassoClassifier(**self.config.model.hyperparameters)
        elif model_name.lower() == 'elasticnet':
            return ElasticNetClassifier(**self.config.model.hyperparameters)
        elif model_name.lower() == 'logistic':
            return LogisticRegressionClassifier(**self.config.model.hyperparameters)
        elif model_name.lower() == 'randomforest':
            return RandomForestClassifier(**self.config.model.hyperparameters)
        elif model_name.lower() == 'catboost':
            return CatBoostClassifier(**self.config.model.hyperparameters)
        elif model_name.lower() == 'neuralnetwork':
            return NeuralNetworkClassifier(**self.config.model.hyperparameters)
        elif model_name.lower() == 'svm':
            return SVMClassifier(**self.config.model.hyperparameters)
        elif model_name.lower() == 'xgboost':
            return XGBoostClassifier(**self.config.model.hyperparameters)
        elif model_name.lower() == 'knn':
            return KNNClassifier(**self.config.model.hyperparameters)
        elif model_name.lower() == 'gaussiannb':
            return GaussianNBClassifier(**self.config.model.hyperparameters)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _evaluate_final_model(self, model, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """评估最终模型。"""
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
        
        # 预测
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary')
        }
        
        # 计算AUC
        if y_pred_proba.shape[1] == 2:
            metrics['auc'] = roc_auc_score(y, y_pred_proba[:, 1])
        else:
            metrics['auc'] = roc_auc_score(y, y_pred_proba, multi_class='ovr')
        
        return metrics
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        使用最终模型进行预测。
        
        Args:
            X: 特征矩阵
            model_name: 模型名称（如果为None，使用第一个模型）
            
        Returns:
            预测结果
        """
        if not self.final_models_:
            raise ValueError("No final models available. Train models first.")
        
        if model_name is None:
            model_name = list(self.final_models_.keys())[0]
        
        if model_name not in self.final_models_:
            raise ValueError(f"Model {model_name} not found in final models")
        
        # 与训练阶段保持一致的预处理（直接复制，不进行额外预处理）
        X_processed = X.copy()
        
        # 选择共识特征
        if self.consensus_features_ is None:
            raise ValueError("Consensus features not set. Train models first.")
        
        consensus_features = self.consensus_features_[model_name]
        X_selected = X_processed[consensus_features]
        
        # 预测
        model = self.final_models_[model_name]
        return model.predict(X_selected)
    
    def predict_proba(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        使用最终模型进行概率预测。
        
        Args:
            X: 特征矩阵
            model_name: 模型名称（如果为None，使用第一个模型）
            
        Returns:
            预测概率
        """
        if not self.final_models_:
            raise ValueError("No final models available. Train models first.")
        
        if model_name is None:
            model_name = list(self.final_models_.keys())[0]
        
        if model_name not in self.final_models_:
            raise ValueError(f"Model {model_name} not found in final models")
        
        # 与训练阶段保持一致的预处理（直接复制，不进行额外预处理）
        X_processed = X.copy()
        
        # 选择共识特征
        if self.consensus_features_ is None:
            raise ValueError("Consensus features not set. Train models first.")
        
        consensus_features = self.consensus_features_[model_name]
        X_selected = X_processed[consensus_features]

        # 预测
        model = self.final_models_[model_name]
        return model.predict_proba(X_selected)
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        获取特征重要性。
        
        Args:
            model_name: 模型名称（如果为None，使用第一个模型）
            
        Returns:
            特征重要性数组
        """
        if not self.final_models_:
            return None
        
        if model_name is None:
            model_name = list(self.final_models_.keys())[0]
        
        if model_name not in self.final_models_:
            return None
        
        model = self.final_models_[model_name]
        return model.get_feature_importance()
    
    def save_final_models(self, output_dir: Union[str, Path]) -> None:
        """
        保存最终模型。
        
        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving final models to {output_dir}")
        
        # 使用标准输出管理器保存最终模型
        for model_name, model in self.final_models_.items():
            self.output_manager.save_final_model(
                output_dir, 
                model, 
                model_name.lower()
            )

        # 将最终阶段参数写入 final_run.yaml（合并已有内容）
        try:
            reproducible_cfg = {
                'final_cv_folds': getattr(self, 'final_cv_folds_used', None),
                'final_search_method': getattr(self, 'final_search_method_used', None),
                'final_cpu': getattr(self, 'final_cpu_used', None)
            }
            
            # 添加自适应参数信息（如果可用）
            if hasattr(self, 'adaptive_param_info_') and self.adaptive_param_info_:
                reproducible_cfg['adaptive_hyperparameter_info'] = self.adaptive_param_info_
            
            # 添加类别平衡信息（如果可用）
            if hasattr(self, 'class_balance_info_') and self.class_balance_info_:
                reproducible_cfg['class_balance_info'] = self.class_balance_info_
            
            # 从自适应参数信息中提取类别标签信息
            if hasattr(self, 'adaptive_param_info_') and self.adaptive_param_info_:
                for model_name, info in self.adaptive_param_info_.items():
                    if 'class_labels' in info:
                        if 'class_balance_info' not in reproducible_cfg:
                            reproducible_cfg['class_balance_info'] = {}
                        reproducible_cfg['class_balance_info'].update(info['class_labels'])
                        break  # 只需要第一个模型的标签信息
            
            # 添加明确的标签映射信息
            if hasattr(self, 'label_mapping_') and self.label_mapping_:
                if 'class_balance_info' not in reproducible_cfg:
                    reproducible_cfg['class_balance_info'] = {}
                reproducible_cfg['class_balance_info']['label_mapping'] = self.label_mapping_
                reproducible_cfg['class_balance_info']['label_0'] = self.label_mapping_.get(0, "Unknown")
                reproducible_cfg['class_balance_info']['label_1'] = self.label_mapping_.get(1, "Unknown")
            
            # 添加外层CV性能指标信息（如果可用）
            if hasattr(self, 'performance_metrics_') and self.performance_metrics_:
                reproducible_cfg['outer_cv_performance_metrics'] = self.performance_metrics_
            
            # 添加共识特征选择统计信息（如果可用）
            if hasattr(self, 'selection_stats_') and self.selection_stats_:
                reproducible_cfg['consensus_feature_selection_stats'] = self.selection_stats_
            
            # 添加外层折汇总信息（如果可用）
            if hasattr(self, 'outer_fold_summary_') and self.outer_fold_summary_:
                reproducible_cfg['outer_fold_summary'] = self.outer_fold_summary_
            
            model_name_for_repro = self.config.model.name.lower() if hasattr(self.config, 'model') and hasattr(self.config.model, 'name') else 'unknown'
            self.output_manager.save_reproducibility_info(output_dir, reproducible_cfg, model_name_for_repro)
        except Exception as e:
            self.logger.warning(f"写入最终阶段参数到final_run.yaml失败: {e}")
        
        # 获取最终模型目录
        final_model_dir = output_dir / "2_final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存共识特征到最终模型目录
        features_path = final_model_dir / "consensus_features.json"
        import json
        with open(features_path, 'w') as f:
            json.dump(self.consensus_features_, f, indent=2)
        
        # 保存增强的训练结果到最终模型目录
        results_path = final_model_dir / "final_training_results.json"
        
        # 构建增强的结果字典
        enhanced_results = {}
        
        # 添加基本信息
        enhanced_results['experiment_info'] = {
            'timestamp': datetime.now().isoformat(),
            'model_count': len(self.final_models_),
            'total_consensus_features': sum(len(features) for features in self.consensus_features_.values()),
            'random_seed': getattr(self.config.cv, 'random_state', 42) if hasattr(self, 'config') and self.config else 42
        }
        
        # 添加配置信息
        if hasattr(self, 'config') and self.config:
            enhanced_results['configuration'] = {
                'cv_strategy': str(getattr(self.config.cv, 'strategy', 'unknown')),
                'outer_folds': getattr(self.config.cv, 'outer_folds', 5),
                'inner_folds': getattr(self.config.cv, 'inner_folds', 3),
                'n_repeats': getattr(self.config.cv, 'n_repeats', 1),
                'n_jobs': getattr(self.config.cv, 'n_jobs', 1),
                'feature_selection_enabled': getattr(self.config.feature_selection, 'enabled', True) if hasattr(self.config, 'feature_selection') else True
            }
        
        # 添加嵌套CV结果摘要
        if hasattr(self, 'nested_cv_results_') and self.nested_cv_results_:
            nested_results = self.nested_cv_results_.get('nested_cv_results', {})
            enhanced_results['nested_cv_summary'] = {
                'accuracy_mean': nested_results.get('accuracy_mean', 0),
                'accuracy_std': nested_results.get('accuracy_std', 0),
                'auc_mean': nested_results.get('auc_mean', 0),
                'auc_std': nested_results.get('auc_std', 0),
                'f1_mean': nested_results.get('f1_mean', 0),
                'f1_std': nested_results.get('f1_std', 0),
                'precision_mean': nested_results.get('precision_mean', 0),
                'precision_std': nested_results.get('precision_std', 0),
                'recall_mean': nested_results.get('recall_mean', 0),
                'recall_std': nested_results.get('recall_std', 0)
            }
        
        # 添加每个模型的详细信息
        enhanced_results['models'] = {}
        for model_name, results in self.training_results_.items():
            model_info = {
                'consensus_features': self.consensus_features_.get(model_name, []),
                'consensus_feature_count': len(self.consensus_features_.get(model_name, [])),
                'final_metrics': results.get('metrics', {}),
                'hyperparameters': results.get('hyperparameters', {}),
                'model_type': type(self.final_models_.get(model_name)).__name__ if model_name in self.final_models_ else 'Unknown'
            }
            
            # 添加模型参数
            if model_name in self.final_models_:
                model = self.final_models_[model_name]
                if hasattr(model, 'get_params'):
                    model_info['sklearn_parameters'] = model.get_params()
            
            enhanced_results['models'][model_name] = model_info
        
        # 保存增强的结果
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
        
        # 生成最终模型报告到最终模型目录
        self._generate_final_model_report(final_model_dir)
        
        # 保存可重现性信息
        if hasattr(self, 'config') and self.config:
            # 获取最终模型的真实超参数
            final_model_hyperparameters = {}
            if self.final_models_:
                for model_name, model in self.final_models_.items():
                    if hasattr(model, 'get_params'):
                        final_model_hyperparameters[model_name] = model.get_params()
                    else:
                        final_model_hyperparameters[model_name] = {}
            
            config_info = {
                'model_name': self.config.model.name.lower() if hasattr(self.config.model, 'name') else 'unknown',
                'cv_strategy': self.config.cv.strategy.value if hasattr(self.config.cv, 'strategy') and hasattr(self.config.cv.strategy, 'value') else (self.config.cv.strategy if hasattr(self.config.cv, 'strategy') else 'unknown'),
                'outer_folds': self.config.cv.outer_folds if hasattr(self.config.cv, 'outer_folds') else 5,
                'inner_folds': self.config.cv.inner_folds if hasattr(self.config.cv, 'inner_folds') else 3,
                'n_repeats': self.config.cv.n_repeats if hasattr(self.config.cv, 'n_repeats') else 1,
                'random_state': self.config.cv.random_state if hasattr(self.config.cv, 'random_state') else 42,
                'n_jobs': self.config.cv.n_jobs if hasattr(self.config.cv, 'n_jobs') else 1,
                'enable_adaptive_filtering': self.config.adaptive_filter.enabled if hasattr(self.config.adaptive_filter, 'enabled') else True,
                'adaptive_filter_config': self.config.adaptive_filter.__dict__ if hasattr(self.config.adaptive_filter, '__dict__') else {},
                'model_hyperparameters': final_model_hyperparameters,  # 使用最终模型的真实参数
                'model_task_type': self.config.model.task_type.value if hasattr(self.config.model, 'task_type') and hasattr(self.config.model.task_type, 'value') else (self.config.model.task_type if hasattr(self.config.model, 'task_type') else 'classification'),
                'feature_selection_config': self.config.feature_selection.__dict__ if hasattr(self.config, 'feature_selection') and self.config.feature_selection else {},
                'data_paths': self.config.data_paths if hasattr(self.config, 'data_paths') and self.config.data_paths else {},
                'preprocessing_config': self.config.model.preprocessing if hasattr(self.config.model, 'preprocessing') and self.config.model.preprocessing else {}
            }
            
            self.output_manager.save_reproducibility_info(
                output_dir, config_info, config_info['model_name']
            )
        
        # 基于评估阶段确定的阈值，在最终模型阶段生成样本级预测与HI
        try:
            decision_thr = None
            # 从final_run.yaml中读回阈值（由StandardOutputManager在评估阶段写入）
            repro_dir = Path(output_dir) / '4_reproducibility'
            final_run_fp = repro_dir / 'final_run.yaml'
            if final_run_fp.exists():
                import yaml
                with open(final_run_fp, 'r', encoding='utf-8') as rf:
                    finfo = yaml.safe_load(rf) or {}
                    cfg = finfo.get('config', {}) if isinstance(finfo, dict) else {}
                    if isinstance(cfg, dict):
                        decision_thr = cfg.get('decision_threshold', None)
            # 若无阈值则跳过
            if decision_thr is not None:
                # 用最终模型对全量数据做预测概率（使用对应模型的共识特征）
                preds_rows = []
                for model_name, model in self.final_models_.items():
                    features = self.consensus_features_.get(model_name, [])
                    # 与训练阶段保持一致的预处理（直接复制，不进行额外预处理）
                    X_processed = self.data_loader.last_X_.copy() if hasattr(self.data_loader, 'last_X_') else None
                    y_true = self.data_loader.last_y_ if hasattr(self.data_loader, 'last_y_') else None
                    if X_processed is None or y_true is None:
                        # 回退：要求调用方提供原始训练数据到本方法上下文；若无则跳过
                        continue
                    X_selected = X_processed[features]
                    proba = model.predict_proba(X_selected)
                    # 兼容二分类概率
                    prob_1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
                    # 组装样本ID（若可获取）
                    if hasattr(self.data_loader, 'last_sample_ids_') and self.data_loader.last_sample_ids_ is not None:
                        sample_ids = list(self.data_loader.last_sample_ids_)
                    else:
                        sample_ids = list(range(1, len(prob_1) + 1))
                    for sid, yt, p1 in zip(sample_ids, y_true, prob_1):
                        preds_rows.append({'sample_id': sid, 'true_value': int(yt), 'prob_1': float(p1)})
                if preds_rows:
                    import pandas as pd
                    dfp = pd.DataFrame(preds_rows)
                    dfp['mgba_hi'] = float(decision_thr) - dfp['prob_1']
                    dfp['threshold'] = float(decision_thr)
                    final_dir = Path(output_dir) / '2_final_model'
                    final_dir.mkdir(parents=True, exist_ok=True)
                    out_fp = final_dir / 'final_predictions.csv'
                    dfp[['sample_id', 'true_value', 'prob_1', 'mgba_hi', 'threshold']].to_csv(out_fp, index=False)
        except Exception as e:
            self.logger.warning(f"生成最终模型阶段的final_predictions.csv失败: {e}")

        self.logger.info("Final models saved successfully")
    
    def load_final_models(self, model_dir: Union[str, Path]) -> 'FinalModelTrainer':
        """
        加载最终模型。
        
        Args:
            model_dir: 模型目录
            
        Returns:
            Self for method chaining
        """
        model_dir = Path(model_dir)
        
        self.logger.info(f"Loading final models from {model_dir}")
        
        # 加载共识特征
        features_path = model_dir / "consensus_features.json"
        if features_path.exists():
            import json
            with open(features_path, 'r') as f:
                self.consensus_features_ = json.load(f)
        
        # 加载模型
        self.final_models_ = {}
        for model_file in model_dir.glob("final_model_*.joblib"):
            model_name = model_file.stem.replace("final_model_", "")
            # 保持模型名称一致性（与训练时保持一致）
            model = joblib.load(model_file)
            self.final_models_[model_name] = model
            self.logger.info(f"  Loaded {model_name}")
        
        # 加载预处理器
        preprocessor_path = model_dir / "data_preprocessor.joblib"
        if preprocessor_path.exists():
            self.data_preprocessor = joblib.load(preprocessor_path)
            self.logger.info("  Loaded data preprocessor")
        
        # 加载自适应方差过滤器
        filter_path = model_dir / "adaptive_variance_filter.joblib"
        if filter_path.exists():
            self.adaptive_filter = joblib.load(filter_path)
            self.logger.info("  Loaded adaptive variance filter")
        
        self.logger.info("Final models loaded successfully")
        return self
    
    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式。"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _generate_final_model_report(self, output_dir: Path):
        """生成最终模型报告。"""
        # 注释掉final_model_report.txt的生成，与JSON重复
        # self.logger.info("Generating final model report...")
        # 
        # report_path = output_dir / "final_model_report.txt"
        # 
        # with open(report_path, 'w') as f:
        #     f.write("最终模型训练报告\n")
        #     f.write("=" * 50 + "\n\n")
        #     
        #     # 模型信息
        #     f.write(f"训练模型数: {len(self.final_models_)}\n")
        #     # 计算总的共识特征数（所有模型的特征总数）
        #     total_features = sum(len(features) for features in self.consensus_features_.values())
        #     f.write(f"共识特征数: {total_features}\n\n")
        #     
        #     # 各模型信息
        #     for model_name, results in self.training_results_.items():
        #         f.write(f"{model_name} 最终模型:\n")
        #         f.write("-" * 30 + "\n")
        #         f.write(f"  共识特征数: {len(results['consensus_features'])}\n")
        #         f.write(f"  准确率: {results['metrics']['accuracy']:.4f}\n")
        #         f.write(f"  AUC: {results['metrics']['auc']:.4f}\n")
        #         f.write(f"  精确率: {results['metrics']['precision']:.4f}\n")
        #         f.write(f"  召回率: {results['metrics']['recall']:.4f}\n")
        #         f.write(f"  F1分数: {results['metrics']['f1']:.4f}\n")
        #         f.write("\n")
        pass
    
    def get_training_results(self) -> Optional[Dict[str, Any]]:
        """获取训练结果。"""
        return self.training_results_
    
    def get_final_models(self) -> Dict[str, Any]:
        """获取最终模型。"""
        return self.final_models_
    
    def get_consensus_features(self) -> Optional[Dict[str, List[str]]]:
        """获取共识特征。"""
        return self.consensus_features_


def create_final_model_trainer(
    model_name: str = "Lasso",
    output_dir: str = "./final_models",
    **kwargs
) -> FinalModelTrainer:
    """
    创建最终模型训练器的工厂函数。
    
    Args:
        model_name: 模型名称
        output_dir: 输出目录
        **kwargs: 其他配置参数
        
    Returns:
        配置好的FinalModelTrainer实例
    """
    from .base import ModelConfig, CVConfig, AdaptiveFilterConfig, ExperimentConfig
    
    # 创建配置
    model_config = ModelConfig(
        name=model_name,
        task_type=TaskType.CLASSIFICATION,
        hyperparameters=kwargs.get('model_params', {})
    )
    
    cv_config = CVConfig(
        strategy=CVStrategy.REPEATED_KFOLD,
        outer_folds=5,
        inner_folds=3,
        random_state=kwargs.get('random_state', 42),
        n_jobs=kwargs.get('n_jobs', 1)
    )
    
    adaptive_filter_config = AdaptiveFilterConfig(
        enabled=kwargs.get('enable_adaptive_filtering', True),
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
        verbose=kwargs.get('verbose', True)
    )
    
    return FinalModelTrainer(experiment_config)
