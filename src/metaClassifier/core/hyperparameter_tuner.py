"""
Hyperparameter tuning strategies for metaClassifier.

This module contains various hyperparameter optimization methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score

from .base import BaseModel
from ..utils.logger import get_logger


class HyperparameterTuner:
    """Main hyperparameter tuner class."""
    
    def __init__(self, method: str = "grid", **kwargs):
        self.method = method
        self.config = kwargs
        self.tuner = None
        
    def tune(
        self, 
        model: BaseModel, 
        X: pd.DataFrame, 
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5
    ) -> BaseModel:
        """Tune hyperparameters for a model."""
        if self.method == "grid":
            self.tuner = GridSearchTuner(**self.config)
        elif self.method == "random":
            self.tuner = RandomizedSearchTuner(**self.config)
        elif self.method in ("bayes",):
            self.tuner = OptunaTuner(**self.config)
        else:
            raise ValueError(f"Unknown hyperparameter tuning method: {self.method}")
        
        return self.tuner.tune(model, X, y, param_grid, cv)


class HyperparameterTunerFactory:
    """Factory for creating hyperparameter tuners."""
    
    @staticmethod
    def create_tuner(method: str, **kwargs) -> 'BaseHyperparameterTuner':
        """Create a hyperparameter tuner based on method name."""
        if method == "grid":
            return GridSearchTuner(**kwargs)
        elif method == "random":
            return RandomizedSearchTuner(**kwargs)
        elif method in ("bayes",):
            return OptunaTuner(**kwargs)
        else:
            raise ValueError(f"Unsupported hyperparameter tuning method: {method}")


class BaseHyperparameterTuner(ABC):
    """Base class for hyperparameter tuners."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.logger = get_logger(self.__class__.__name__)
        
    @abstractmethod
    def tune(
        self, 
        model: BaseModel, 
        X: pd.DataFrame, 
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5
    ) -> BaseModel:
        """Tune hyperparameters for a model."""
        pass


class GridSearchTuner(BaseHyperparameterTuner):
    """Grid search hyperparameter tuner."""
    
    def __init__(self, scoring: str = "roc_auc", n_jobs: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.scoring = scoring
        self.n_jobs = n_jobs
        
    def tune(
        self, 
        model: BaseModel, 
        X: pd.DataFrame, 
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5
    ) -> BaseModel:
        """Tune hyperparameters using grid search."""
        self.logger.info(f"GridSearch | scoring=combined_binary | cv={cv} | n_jobs={self.n_jobs}")
        # 针对二分类任务优化的综合评分器
        def binary_combined_scorer(estimator, X, y):
            """二分类任务的综合评分器（ROC-AUC + 平均精度）"""
            try:
                y_pred_proba = estimator.predict_proba(X)
                if y_pred_proba.shape[1] == 2:
                    y_proba = y_pred_proba[:, 1]
                else:
                    y_proba = y_pred_proba[:, 0]
                
                # 计算多个指标的综合得分
                roc_auc = roc_auc_score(y, y_proba)
                avg_precision = average_precision_score(y, y_proba)
                
                # 综合得分：ROC-AUC权重0.7，平均精度权重0.3
                combined_score = 0.7 * roc_auc + 0.3 * avg_precision
                return combined_score
            except Exception:
                return 0.0
        
        # 使用自定义综合评分器
        scorer = binary_combined_scorer
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scorer,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.cv_results_ = grid_search.cv_results_
        self.logger.info(f"GridSearch | best_score={self.best_score_} | best_params={self.best_params_}")
        
        # Update model with best parameters
        model.set_params(**self.best_params_)
        model.fit(X, y)
        
        return model


class RandomizedSearchTuner(BaseHyperparameterTuner):
    """Randomized search hyperparameter tuner."""
    
    def __init__(self, n_iter: int = 100, scoring: str = "roc_auc", n_jobs: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        
    def tune(
        self, 
        model: BaseModel, 
        X: pd.DataFrame, 
        y: np.ndarray,
        param_distributions: Dict[str, List[Any]],
        cv: int = 5
    ) -> BaseModel:
        """Tune hyperparameters using randomized search."""
        self.logger.info(f"RandomSearch | scoring=roc_auc | cv={cv} | n_iter={self.n_iter} | n_jobs={self.n_jobs} | seed=42")
        # 针对二分类任务优化的自定义scorer
        def binary_roc_auc_scorer(estimator, X, y):
            """二分类任务的ROC AUC评分器"""
            try:
                y_pred_proba = estimator.predict_proba(X)
                if y_pred_proba.shape[1] == 2:
                    return roc_auc_score(y, y_pred_proba[:, 1])
                else:
                    return roc_auc_score(y, y_pred_proba[:, 0])
            except Exception:
                return 0.0
        # 使用内置的roc_auc评分器，兼容新版本scikit-learn
        scorer = 'roc_auc'
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=cv,
            scoring=scorer,
            n_jobs=self.n_jobs,
            verbose=0,
            random_state=42
        )
        
        random_search.fit(X, y)
        
        self.best_params_ = random_search.best_params_
        self.best_score_ = random_search.best_score_
        self.cv_results_ = random_search.cv_results_
        self.logger.info(f"RandomSearch | best_score={self.best_score_} | best_params={self.best_params_}")
        
        # Update model with best parameters
        model.set_params(**self.best_params_)
        model.fit(X, y)
        
        return model


# 说明：bayes/bayesian 模式统一映射到 OptunaTuner


class OptunaTuner(BaseHyperparameterTuner):
    """Optuna TPE-based hyperparameter tuner with pruning support."""

    def __init__(self, n_trials: int = 100, scoring: str = "roc_auc", n_jobs: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.n_trials = n_trials
        self.scoring = scoring
        self.n_jobs = n_jobs
        self._optuna_available = self._check_optuna_availability()

    def _check_optuna_availability(self) -> bool:
        """检查optuna是否可用并记录版本信息"""
        try:
            import optuna
            # 检查版本兼容性
            version = getattr(optuna, '__version__', 'unknown')
            self.logger.info(f"Optuna版本: {version}")
            return True
        except ImportError:
            self.logger.warning("Optuna未安装，贝叶斯调参将不可用")
            return False
        except Exception as e:
            self.logger.warning(f"Optuna检查失败: {e}")
            return False

    def _build_objective(self, model: BaseModel, X: pd.DataFrame, y: np.ndarray, param_grid: Dict[str, List[Any]], cv: int):
        import optuna
        
        # 使用内置的roc_auc评分器，兼容新版本scikit-learn
        scoring = 'roc_auc'
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Normalize search space: use appropriate suggestion methods based on parameter type
        def suggest_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
            suggested = {}
            for name, values in (param_grid or {}).items():
                # 处理范围格式 [min, max] 和列表格式
                if isinstance(values, (list, tuple)) and len(values) == 2:
                    # 范围格式 [min, max]
                    min_val, max_val = values
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                        # 根据参数类型选择合适的建议方法
                        if self._is_log_scale_param(name, [min_val, max_val]):
                            suggested[name] = trial.suggest_float(name, min_val, max_val, log=True)
                        elif self._is_integer_param(name, [min_val, max_val]):
                            suggested[name] = trial.suggest_int(name, int(min_val), int(max_val))
                        else:
                            suggested[name] = trial.suggest_float(name, min_val, max_val, log=False)
                        continue
                
                # 处理列表格式
                try:
                    candidates = list(values)
                except Exception:
                    candidates = [values]
                if not candidates:
                    continue
                
                # 更智能的参数类型判断
                if self._is_log_scale_param(name, candidates):
                    # 对数尺度参数（如C, alpha, gamma, learning_rate）
                    # 过滤掉 None 值
                    numeric_candidates = [c for c in candidates if c is not None and isinstance(c, (int, float))]
                    if not numeric_candidates:
                        suggested[name] = trial.suggest_categorical(name, candidates)
                    else:
                        min_val = min(numeric_candidates)
                        max_val = max(numeric_candidates)
                        if min_val > 0 and max_val > min_val:
                            suggested[name] = trial.suggest_float(name, min_val, max_val, log=True)
                        else:
                            suggested[name] = trial.suggest_categorical(name, candidates)
                elif self._is_integer_param(name, candidates):
                    # 整数参数
                    # 过滤掉 None 值
                    int_candidates = [c for c in candidates if c is not None]
                    if not int_candidates:
                        suggested[name] = trial.suggest_categorical(name, candidates)
                    else:
                        min_val = int(min(int_candidates))
                        max_val = int(max(int_candidates))
                        suggested[name] = trial.suggest_int(name, min_val, max_val)
                elif self._is_categorical_param(name, candidates):
                    # 分类参数（字符串、布尔值等）
                    suggested[name] = trial.suggest_categorical(name, candidates)
                else:
                    # 默认使用线性尺度的浮点数
                    # 过滤掉 None 值
                    numeric_candidates = [c for c in candidates if c is not None and isinstance(c, (int, float))]
                    if not numeric_candidates:
                        suggested[name] = trial.suggest_categorical(name, candidates)
                    else:
                        min_val = min(numeric_candidates)
                        max_val = max(numeric_candidates)
                        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                            suggested[name] = trial.suggest_float(name, min_val, max_val, log=False)
                        else:
                            suggested[name] = trial.suggest_categorical(name, candidates)
            return suggested

        def objective(trial: optuna.trial.Trial) -> float:
            params = suggest_params(trial)
            try:
                # 设置参数并进行交叉验证评分（AUC）
                model.set_params(**params)
            except (ValueError, TypeError) as e:
                # 参数不兼容时给出较差分数，但不记录为警告（这是正常的优化过程）
                self.logger.debug(f"参数设置失败，跳过此参数组合: {e}")
                return 0.0
            except Exception as e:
                # 其他异常才记录为警告
                self.logger.warning(f"参数设置出现意外错误: {e}")
                return 0.0
            try:
                scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=self.n_jobs)
                if len(scores) == 0:
                    self.logger.warning("交叉验证返回空分数")
                    return 0.0
                mean_score = float(scores.mean())
                if np.isnan(mean_score) or np.isinf(mean_score):
                    self.logger.debug(f"分数异常，跳过此参数组合: {mean_score}")
                    return 0.0
                return mean_score
            except Exception as e:
                self.logger.warning(f"交叉验证计算失败: {e}")
                return 0.0

        return objective

    def _is_log_scale_param(self, name: str, candidates: List[Any]) -> bool:
        """判断参数是否适合对数尺度优化"""
        log_scale_params = {
            'C', 'alpha', 'gamma', 'learning_rate', 'l2_leaf_reg', 'reg_alpha', 'reg_lambda'
        }
        return name in log_scale_params

    def _is_integer_param(self, name: str, candidates: List[Any]) -> bool:
        """判断参数是否为整数类型"""
        integer_params = {
            'max_iter', 'n_estimators', 'n_neighbors', 'n_jobs', 'max_depth', 
            'min_samples_split', 'min_samples_leaf', 'depth', 'iterations'
        }
        if name in integer_params:
            return True
        # 检查所有候选值是否都是整数
        return all(isinstance(x, int) for x in candidates)

    def _is_categorical_param(self, name: str, candidates: List[Any]) -> bool:
        """判断参数是否为分类类型"""
        categorical_params = {
            'kernel', 'weights', 'metric', 'activation', 'hidden_layer_sizes', 'solver', 'penalty'
        }
        if name in categorical_params:
            return True
        # 检查是否包含非数值类型
        return any(not isinstance(x, (int, float)) for x in candidates)

    def tune(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5
    ) -> BaseModel:
        if not self._optuna_available:
            raise ImportError(
                "Optuna-based search requires optuna. Please install it, e.g. `pip install optuna`, "
                "or switch --final_search_method to 'grid' or 'random'."
            )
        import optuna
        
        # 尝试获取贝叶斯优化的参数网格
        bayesian_param_grid = self._get_bayesian_param_grid(model, param_grid)
        if bayesian_param_grid:
            self.logger.info("使用贝叶斯优化专用参数网格")
            param_grid = bayesian_param_grid
        
        # 固定随机性
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        # 日志更清晰：区分 trial 间并行(optuna_n_jobs) 与 trial 内CV并行(cv_n_jobs)
        optuna_n_jobs = 1  # study.optimize 固定单trial并行
        cv_n_jobs = self.n_jobs
        self.logger.info(
            f"Optuna | scoring=roc_auc | cv={cv} | trials={self.n_trials} | optuna_n_jobs={optuna_n_jobs} | cv_n_jobs={cv_n_jobs} | seed=42 | pruner=Median"
        )
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        objective = self._build_objective(model, X, y, param_grid, cv)
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)  # 由内部 cross_val_score 控制并行

        self.best_params_ = study.best_trial.params if study.best_trial else {}
        self.best_score_ = study.best_value if study.best_trial else None
        self.cv_results_ = None  # 可选：导出 trial 历史
        self.logger.info(f"Optuna | best_score={self.best_score_} | best_params={self.best_params_}")

        # 设置最佳参数并在全量数据上拟合
        if self.best_params_:
            model.set_params(**self.best_params_)
        model.fit(X, y)
        return model
    
    def _get_bayesian_param_grid(self, model: BaseModel, param_grid: Dict[str, List[Any]]) -> Optional[Dict[str, List[Any]]]:
        """获取贝叶斯优化的参数网格"""
        try:
            from .model_specific_hyperparameter_tuner import ModelSpecificHyperparameterTuner
            ms_tuner = ModelSpecificHyperparameterTuner()
            model_name = getattr(model, 'name', '').lower()
            if model_name:
                bayesian_grid = ms_tuner.get_bayesian_param_grid(model_name)
                if bayesian_grid:
                    return bayesian_grid
        except Exception as e:
            self.logger.debug(f"无法获取贝叶斯参数网格: {e}")
        return None
    
    
