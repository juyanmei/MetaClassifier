"""
Extended Final-stage CV Evaluator (补救方案)

用途：在第二阶段（最终模型阶段）导出“参数调优所用的CV结果”，包括：
- 每折的OOF预测概率（prob_1）
- 每折与总体的AUC、Accuracy、Precision、Recall、F1
- 若可用，则导出调参器的 cv_results_

说明：
- 本扩展不修改主程序逻辑，独立运行。
- 预期输入为：第一阶段得到的共识特征（或文件路径）、模型名称、搜索方法、CV折数，以及原始特征矩阵X与标签y。
- 预处理对齐最终阶段：不再做标准化/缩放，自适应过滤已在第一阶段完成，只对共识特征做子集选择。
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)

from ..core.hyperparameter_tuner import HyperparameterTuner
from ..core.model_specific_hyperparameter_tuner import ModelSpecificHyperparameterTuner


class ExtendedFinalCVEvaluator:
    """
    第二阶段CV导出器（独立、补救型）。
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        final_cv_folds: int = 5,
        final_search_method: str = "grid",
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_cv_folds = final_cv_folds
        self.final_search_method = final_search_method
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _map_method(self, method: str) -> str:
        mapping = {
            "grid": "grid",
            "random": "random",
            "bayes": "bayes",
        }
        return mapping.get(method, "grid")

    def _create_base_model(self, model_name: str):
        from ..models import (
            LassoClassifier, RandomForestClassifier, NeuralNetworkClassifier,
            ElasticNetClassifier, LogisticRegressionClassifier, CatBoostClassifier,
            SVMClassifier, XGBoostClassifier, KNNClassifier, GaussianNBClassifier
        )
        name = (model_name or "").lower()
        if name == "lasso":
            return LassoClassifier()
        if name == "elasticnet":
            return ElasticNetClassifier()
        if name == "logistic":
            return LogisticRegressionClassifier()
        if name == "randomforest":
            return RandomForestClassifier()
        if name == "catboost":
            return CatBoostClassifier()
        if name == "neuralnetwork":
            return NeuralNetworkClassifier()
        if name == "svm":
            return SVMClassifier()
        if name == "xgboost":
            return XGBoostClassifier()
        if name == "knn":
            return KNNClassifier()
        if name == "gaussiannb":
            return GaussianNBClassifier()
        raise ValueError(f"Unknown model: {model_name}")

    def run(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
        consensus_features: Optional[List[str]] = None,
        consensus_features_path: Optional[Union[str, Path]] = None,
        model_alias_for_output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行第二阶段的“选参CV复刻 + OOF导出”。

        Inputs:
        - X, y: 原始特征与标签（与最终阶段一致，不再进行额外缩放）
        - model_name: 模型名称
        - consensus_features: 第一阶段得到的共识特征列表（优先使用）
        - consensus_features_path: 共识特征JSON路径（如提供则可从中取出 model_name 对应特征）
        - model_alias_for_output: 输出文件名中的模型别名（可选）
        """

        # 解析特征
        features = list(consensus_features or [])
        if not features and consensus_features_path:
            with open(consensus_features_path, "r", encoding="utf-8") as f:
                feat_map = json.load(f)
            if isinstance(feat_map, dict):
                # 优先精确匹配；若无则做不区分大小写的键匹配
                features = feat_map.get(model_name, [])
                if not features:
                    lower_key_map = {str(k).lower(): v for k, v in feat_map.items()}
                    features = lower_key_map.get((model_name or "").lower(), [])
        if not features:
            raise ValueError("共识特征为空：请提供 consensus_features 或 consensus_features_path")

        name_for_out = (model_alias_for_output or model_name or "model").lower()

        # 子集选择（与最终阶段一致）
        X_selected = X[features].copy()

        # 构建参数网格
        ms_tuner = ModelSpecificHyperparameterTuner()
        if self.final_search_method == "bayes":
            param_grid = ms_tuner.get_bayesian_param_grid(model_name)
        else:
            param_grid = ms_tuner.get_param_grid(model_name)

        # 调参（复用与主程序相同的 HyperparameterTuner）
        base_model = self._create_base_model(model_name)
        tuner = HyperparameterTuner(method=self._map_method(self.final_search_method), n_jobs=self.n_jobs)
        tuned_model = tuner.tune(base_model, X_selected, y, param_grid, cv=self.final_cv_folds)
        best_params = getattr(tuner.tuner, "best_params_", {}) if hasattr(tuner, "tuner") else {}
        cv_results = getattr(tuner.tuner, "cv_results_", None) if hasattr(tuner, "tuner") else None

        # 保存调参器cv_results_
        if cv_results is not None:
            cv_df = pd.DataFrame(cv_results)
            cv_out = self.output_dir / f"cv_results_{name_for_out}.csv"
            cv_df.to_csv(cv_out, index=False)

        # 使用最佳参数进行手动CV以导出OOF概率
        skf = StratifiedKFold(n_splits=self.final_cv_folds, shuffle=True, random_state=self.random_state)
        oof_rows: List[Dict[str, Any]] = []
        fold_metrics: List[Dict[str, Any]] = []

        # 稳定sample_id（若索引存在则使用索引）
        sample_ids = list(X_selected.index) if hasattr(X_selected, "index") else list(range(len(X_selected)))
        y = np.asarray(y)

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_selected, y), start=1):
            X_tr, X_va = X_selected.iloc[tr_idx], X_selected.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # 构建最佳参数模型
            model = self._create_base_model(model_name)
            try:
                if best_params:
                    model.set_params(**best_params)
            except Exception:
                pass

            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                prob_1 = proba[:, 1]
            else:
                prob_1 = proba.reshape(-1)
            y_pred = (prob_1 >= 0.5).astype(int)

            # 计算指标
            try:
                auc = roc_auc_score(y_va, prob_1)
            except Exception:
                auc = float("nan")
            acc = accuracy_score(y_va, y_pred)
            pre = precision_score(y_va, y_pred, average="binary")
            rec = recall_score(y_va, y_pred, average="binary")
            f1 = f1_score(y_va, y_pred, average="binary")

            fold_metrics.append({
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "auc": float(auc) if np.isfinite(auc) else None,
                "accuracy": float(acc),
                "precision": float(pre),
                "recall": float(rec),
                "f1": float(f1),
            })

            # 记录OOF
            for sid, yt, p1 in zip([sample_ids[i] for i in va_idx], y_va, prob_1):
                oof_rows.append({
                    "sample_id": sid,
                    "fold": fold_idx,
                    "true_value": int(yt),
                    "prob_1": float(p1),
                })

        # 汇总指标
        auc_values = [m["auc"] for m in fold_metrics if m.get("auc") is not None]
        
        # 基于所有OOF预测概率重新计算总体AUC（更准确）
        oof_df = pd.DataFrame(oof_rows)
        auc_combined = None
        if not oof_df.empty and 'true_value' in oof_df.columns and 'prob_1' in oof_df.columns:
            y_true_all = oof_df['true_value'].values
            y_prob_all = oof_df['prob_1'].values
            try:
                auc_combined = float(roc_auc_score(y_true_all, y_prob_all))
            except Exception:
                auc_combined = None
        
        summary = {
            "final_search_method": self.final_search_method,
            "final_cv_folds": self.final_cv_folds,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "model_name": model_name,
            "best_params": best_params,
            "auc_mean": float(np.mean(auc_values)) if auc_values else None,  # 每折AUC的均值（参考）
            "auc_std": float(np.std(auc_values)) if auc_values else None,  # 每折AUC的标准差
            "auc_combined": auc_combined,  # 基于所有OOF概率计算的总体AUC（主要指标）
        }

        # 写盘
        oof_out = self.output_dir / f"oof_predictions_{name_for_out}.csv"
        oof_df.to_csv(oof_out, index=False)

        folds_out = self.output_dir / f"fold_metrics_{name_for_out}.json"
        with open(folds_out, "w", encoding="utf-8") as f:
            json.dump(fold_metrics, f, indent=2, ensure_ascii=False)

        summary_out = self.output_dir / f"summary_{name_for_out}.json"
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return {
            "oof_path": str(oof_out),
            "cv_results_path": str(self.output_dir / f"cv_results_{name_for_out}.csv") if cv_results is not None else None,
            "fold_metrics_path": str(folds_out),
            "summary_path": str(summary_out),
        }


def run_extended_final_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    output_root: Union[str, Path],
    model_name: str,
    consensus_features: Optional[List[str]] = None,
    consensus_features_path: Optional[Union[str, Path]] = None,
    final_cv_folds: int = 5,
    final_search_method: str = "grid",
    n_jobs: int = 1,
    model_alias_for_output: Optional[str] = None,
) -> Dict[str, Any]:
    """
    便捷入口函数。
    输出目录固定为 {output_root}/2_final_model/extended_final_cv/。
    """
    output_dir = Path(output_root) / "2_final_model" / "extended_final_cv"
    evaluator = ExtendedFinalCVEvaluator(
        output_dir=output_dir,
        final_cv_folds=final_cv_folds,
        final_search_method=final_search_method,
        random_state=42,
        n_jobs=n_jobs,
    )
    return evaluator.run(
        X=X,
        y=y,
        model_name=model_name,
        consensus_features=consensus_features,
        consensus_features_path=consensus_features_path,
        model_alias_for_output=model_alias_for_output,
    )


