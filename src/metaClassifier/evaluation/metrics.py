"""
Evaluation metrics for metaClassifier.

This module contains various evaluation metrics for classification tasks.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

from ..utils.logger import get_logger


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    def __init__(self):
        self.logger = get_logger("MetricsCalculator")
        
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics for binary classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # 确保是二分类
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            raise ValueError("This metrics calculator is designed for binary classification")
        
        # 将标签转换为0和1
        y_true_binary = np.where(y_true == unique_labels[1], 1, 0)
        y_pred_binary = np.where(y_pred == unique_labels[1], 1, 0)
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # 二分类专用指标
        metrics['specificity'] = self._calculate_specificity(y_true_binary, y_pred_binary)
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        metrics['matthews_corrcoef'] = self._calculate_mcc(y_true_binary, y_pred_binary)
        
        # Probability-based metrics
        if y_proba is not None:
            if y_proba.shape[1] == 2:
                # 二分类：使用正类概率
                y_proba_binary = y_proba[:, 1]
            else:
                # 如果只有一列概率，直接使用
                y_proba_binary = y_proba.flatten()
            
            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_proba_binary)
            metrics['average_precision'] = average_precision_score(y_true_binary, y_proba_binary)
            metrics['log_loss'] = self._calculate_log_loss(y_true_binary, y_proba_binary)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 类别分布信息
        metrics['class_distribution'] = {
            'positive_class': unique_labels[1],  # 通常是疾病组
            'negative_class': unique_labels[0],  # 通常是健康组
            'positive_count': int(np.sum(y_true_binary)),
            'negative_count': int(len(y_true_binary) - np.sum(y_true_binary)),
            'class_balance': float(np.sum(y_true_binary) / len(y_true_binary))
        }
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算特异性（真阴性率）。"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tn + fp == 0:
            return 0.0
        return tn / (tn + fp)
    
    def _calculate_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算Matthews相关系数。"""
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(y_true, y_pred)
    
    def _calculate_log_loss(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """计算对数损失。"""
        from sklearn.metrics import log_loss
        # 确保概率在[0,1]范围内
        y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
        return log_loss(y_true, y_proba)
        
        return metrics
        
    def calculate_cv_metrics(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics from cross-validation results.
        
        Args:
            cv_results: Cross-validation results dictionary
            
        Returns:
            Dictionary of aggregated metrics
        """
        self.logger.info("Calculating CV metrics...")
        
        # Extract all fold results
        fold_results = cv_results.get('fold_results', [])
        
        if not fold_results:
            self.logger.warning("No fold results found in CV results")
            return {}
        
        # Aggregate metrics across folds
        metrics = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric_name in metric_names:
            values = []
            for fold_result in fold_results:
                if metric_name in fold_result:
                    values.append(fold_result[metric_name])
            
            if values:
                metrics[f'{metric_name}_mean'] = np.mean(values)
                metrics[f'{metric_name}_std'] = np.std(values)
                metrics[f'{metric_name}_values'] = values
        
        # Calculate additional aggregated metrics
        metrics['n_folds'] = len(fold_results)
        metrics['total_samples'] = sum(fold_result.get('n_samples', 0) for fold_result in fold_results)
        
        self.logger.info("CV metrics calculated successfully")
        return metrics
        
    def _calculate_specificity(self, confusion_matrix: np.ndarray) -> float:
        """Calculate specificity from confusion matrix."""
        if confusion_matrix.shape != (2, 2):
            return 0.0
            
        tn, fp, fn, tp = confusion_matrix.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
    def generate_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Generate a detailed classification report."""
        return classification_report(
            y_true, y_pred, 
            target_names=target_names,
            output_dict=False
        )
