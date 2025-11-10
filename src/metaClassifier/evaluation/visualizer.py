"""
Visualization utilities for metaClassifier.

This module contains various visualization tools for results and analysis.
"""

from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.logger import get_logger


class ResultsVisualizer:
    """Visualizer for metaClassifier results."""
    
    def __init__(self, style: str = "whitegrid", figsize: tuple = (10, 8)):
        self.style = style
        self.figsize = figsize
        self.logger = get_logger("ResultsVisualizer")
        
        # Set plotting style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        
    def _detect_label_proba_columns(self, df: pd.DataFrame,
                                    label_col: Optional[str] = None,
                                    prob_col: Optional[str] = None) -> Optional[tuple]:
        """统一识别标签列与正类概率列名，返回 (label_col, prob_col) 或 None。"""
        if label_col is None:
            for c in ['y_true', 'true_label', 'label', 'y', 'Y']:
                if c in df.columns:
                    label_col = c; break
        if prob_col is None:
            for c in ['prob_1', 'y_proba', 'predicted_score', 'proba', 'score']:
                if c in df.columns:
                    prob_col = c; break
        if label_col is None or prob_col is None:
            return None
        return (label_col, prob_col)

    def _setup_roc_axes(self,
                         figsize: tuple = (6.0, 6.0),
                         title: Optional[str] = None,
                         xlab: str = '1 - Specificity',
                         ylab: str = 'Sensitivity',
                         show_random: bool = True,
                         legend_loc: str = 'lower right') -> None:
        """统一创建ROC画布与坐标轴样式。"""
        plt.figure(figsize=figsize)
        if show_random:
            plt.plot([0, 1], [0, 1], color='navy', lw=1.2, linestyle='--', alpha=0.6, label='Random')
        if title:
            plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid(False)
        # 注意：图例在调用者绘制主曲线后再触发 plt.legend
        
    def plot_cv_metrics(self, 
                       cv_results: Dict[str, Any], 
                       save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot cross-validation metrics in a single plot."""
        self.logger.info("Creating CV metrics plot...")
        
        # Extract metrics
        metrics = cv_results.get('metrics', {})
        
        if not metrics:
            self.logger.warning("No metrics found in CV results")
            return
        
        # Prepare data for plotting
        metric_names = []
        metric_means = []
        metric_stds = []
        
        # Define metrics in order of importance
        metric_order = ['accuracy', 'auc', 'f1', 'precision', 'recall']
        metric_labels = ['Accuracy', 'ROC AUC', 'F1 Score', 'Precision', 'Recall']
        
        for metric, label in zip(metric_order, metric_labels):
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            
            if mean_key in metrics:
                metric_names.append(label)
                metric_means.append(metrics[mean_key])
                metric_stds.append(metrics.get(std_key, 0))
        
        if not metric_names:
            self.logger.warning("No valid metrics found for plotting")
            return
        
        # Create single plot
        plt.figure(figsize=(12, 8))
        
        # Create bar plot with error bars
        bars = plt.bar(metric_names, metric_means, 
                      yerr=metric_stds, 
                      capsize=5, 
                      alpha=0.7,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(metric_names)])
        
        # Customize plot
        plt.title('Cross-Validation Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on top of bars
        for i, (bar, mean, std) in enumerate(zip(bars, metric_means, metric_stds)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add horizontal line at 0.5 (random performance)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Performance')
        
        # Add horizontal line at 0.8 (good performance)
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good Performance')
        
        # Customize grid
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_feature_importance(self, 
                               feature_importance: np.ndarray,
                               feature_names: List[str],
                               top_k: int = 20,
                               save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot feature importance."""
        self.logger.info("Creating feature importance plot...")
        
        # Get top k features
        top_indices = np.argsort(feature_importance)[-top_k:]
        top_importance = feature_importance[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Create plot
        plt.figure(figsize=self.figsize)
        sns.barplot(x=top_importance, y=top_names)
        plt.title(f'Top {top_k} Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray,
                             class_names: List[str] = None,
                             save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot confusion matrix."""
        self.logger.info("Creating confusion matrix plot...")
        
        if class_names is None:
            class_names = ['Class 0', 'Class 1']
        
        plt.figure(figsize=(5.0, 5.5))
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
        
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_proba: np.ndarray,
                      save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        self.logger.info("Creating ROC curve plot...")
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create plot (统一样式)
        self._setup_roc_axes(figsize=(6.0, 6.0), title='Receiver Operating Characteristic (ROC) Curve',
                             xlab='1 - Specificity', ylab='Sensitivity', show_random=True)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curve plot saved to {save_path}")
        
        plt.show()

    def plot_repeat_mean_roc_from_nested_csv(self,
                                             csv_path: Union[str, Path],
                                             disease: Optional[str] = None,
                                             model: Optional[str] = None,
                                             save_path: Optional[Union[str, Path]] = None,
                                             prob_col: Optional[str] = None,
                                             label_col: Optional[str] = None,
                                             repeat_col: str = 'repeat',
                                             outer_col: str = 'outer_fold',
                                             show_individual: bool = False,
                                             ci_mode: str = 'percentile',  # 'percentile' or 'std'
                                             ci_level: float = 0.95,
                                             fpr_step: float = 0.001,
                                             line_width: float = 2.2,
                                             mean_color: str = '#d62728',
                                             band_alpha: float = 0.2,
                                             grid_alpha: float = 0.25) -> None:
        """基于 nested_cv_pred_proba.csv（包含 repeat 和 outer_fold）绘制均值ROC曲线与置信带。

        过程：
          1) 读取 CSV，识别真实标签列与正类概率列（prob_1）。
          2) 按 repeat 聚合（合并该 repeat 的全部 outer_fold OOF 预测），计算每个 repeat 的 ROC 与 AUC。
          3) 将各 repeat 的 TPR 插值到统一 FPR 网格（0..1, 步长 fpr_step），对 TPR 取均值与区间；AUC 取均值与std或分位数。

        参数：
          - csv_path: nested_cv_pred_proba.csv 路径。
          - disease, model: 用于标题标注。
          - prob_col, label_col: 可显式指定列名；不指定则自动探测。
          - repeat_col, outer_col: 列名（存在即可，不强制使用 outer_col）。
          - show_individual: 是否叠加绘制各 repeat ROC（透明细线）。
          - ci_mode: 'percentile' 使用分位数带；'std' 使用均值±std 带。
          - ci_level: 置信水平（默认95%）。
        """
        import os as _os
        from sklearn.metrics import roc_curve as _roc_curve, auc as _auc

        csv_path = Path(csv_path)
        if not csv_path.exists():
            self.logger.warning(f"nested_cv_pred_proba.csv 不存在: {csv_path}")
            return

        try:
            df = pd.read_csv(str(csv_path))
        except Exception as e:
            self.logger.warning(f"读取CSV失败: {csv_path} | {e}")
            return

        # 自动探测列名（复用工具）
        cols = self._detect_label_proba_columns(df, label_col, prob_col)
        if cols is None:
            self.logger.warning(f"无法识别标签/概率列: label_col={label_col}, prob_col={prob_col}")
            return
        label_col, prob_col = cols

        # 清洗数据
        sub = df[[label_col, prob_col] + ([repeat_col] if repeat_col in df.columns else [])].copy()
        sub = sub.dropna(subset=[label_col, prob_col])
        # 规范标签到 {0,1}
        try:
            sub[label_col] = sub[label_col].astype(float)
        except Exception:
            # 尝试字符串映射
            sub[label_col] = sub[label_col].astype(str).str.strip()
            uniques = sorted(sub[label_col].unique())
            if len(uniques) == 2:
                mapping = {uniques[0]: 0, uniques[1]: 1}
                sub[label_col] = sub[label_col].map(mapping)
        # 过滤非法
        sub = sub[(sub[label_col] == 0) | (sub[label_col] == 1)]
        if sub.empty:
            self.logger.warning("清洗后无有效样本用于ROC计算")
            return

        # 统一的FPR网格
        fpr_grid = np.arange(0.0, 1.0 + 1e-12, float(fpr_step))

        # 按 repeat 计算 ROC
        if repeat_col in sub.columns:
            repeats = sorted([r for r in sub[repeat_col].dropna().unique().tolist()])
        else:
            repeats = [None]
        tpr_grids = []
        auc_list = []

        plt.figure(figsize=(6.0, 6))
        # 统一创建画布与坐标轴
        title_parts = ["Repeat-mean ROC"]
        if disease:
            title_parts.append(f"Disease={disease}")
        if model:
            title_parts.append(f"Model={model}")
        title_str = " | ".join(title_parts) if title_parts else None
        self._setup_roc_axes(figsize=(6.0, 6.0), title=title_str,
                             xlab='1 - Specificity', ylab='Sensitivity', show_random=True)
        if show_individual and len(repeats) > 1:
            # 底色淡线绘制各repeat
            for r in repeats:
                sdf = sub[sub[repeat_col] == r] if r is not None else sub
                y = sdf[label_col].values
                p = sdf[prob_col].values
                if len(np.unique(y)) < 2:
                    continue
                fpr, tpr, _ = _roc_curve(y, p)
                try:
                    plt.plot(fpr, tpr, color='gray', alpha=0.25, lw=1.0)
                except Exception:
                    pass

        # 真正用于统计的再来一次，生成均值+区间所需数据
        # 关键修复：按repeat聚合所有outer_fold的数据，重新计算每个repeat的整体AUC
        for r in repeats:
            # 获取该repeat的所有数据（包括所有outer_fold）
            sdf = sub[sub[repeat_col] == r] if r is not None else sub
            if sdf.empty:
                continue
            
            # 提取该repeat的所有样本的真实标签和预测概率
            y = sdf[label_col].values
            p = sdf[prob_col].values
            
            # 检查是否有足够的类别
            if len(np.unique(y)) < 2:
                self.logger.warning(f"Repeat {r} 中类别数不足，跳过")
                continue
            
            # 重新计算该repeat的整体ROC曲线和AUC（基于所有outer_fold的OOF预测）
            fpr, tpr, _ = _roc_curve(y, p)
            # 插值到统一网格
            tpr_i = np.interp(fpr_grid, fpr, tpr, left=0.0, right=1.0)
            tpr_grids.append(tpr_i)
            # 重新计算该repeat的整体AUC（不是从文件读取，而是基于所有outer_fold的OOF预测重新计算）
            auc_repeat = float(_auc(fpr, tpr))
            auc_list.append(auc_repeat)
            self.logger.debug(f"Repeat {r}: AUC = {auc_repeat:.4f} (基于 {len(y)} 个样本重新计算)")

        if not tpr_grids:
            self.logger.warning("所有 repeat 中均无法计算有效 ROC（可能类别单一），终止绘图")
            return

        tpr_mat = np.vstack(tpr_grids)
        tpr_mean = tpr_mat.mean(axis=0)
        if ci_mode.lower() == 'std' or len(tpr_grids) == 1:
            tpr_std = tpr_mat.std(axis=0)
            tpr_lo, tpr_hi = np.clip(tpr_mean - tpr_std, 0.0, 1.0), np.clip(tpr_mean + tpr_std, 0.0, 1.0)
            ci_text = '±1 SD'
        else:
            alpha = (1.0 - float(ci_level)) / 2.0
            lo_q = 100 * alpha
            hi_q = 100 * (1.0 - alpha)
            tpr_lo = np.percentile(tpr_mat, lo_q, axis=0)
            tpr_hi = np.percentile(tpr_mat, hi_q, axis=0)
            ci_text = f"{int(100*ci_level)}% PI"

        # 绘制均值曲线与置信带
        plt.plot(fpr_grid, tpr_mean, color=mean_color, lw=line_width, label='Mean ROC')
        try:
            plt.fill_between(fpr_grid, tpr_lo, tpr_hi, color=mean_color, alpha=band_alpha, label=ci_text)
        except Exception:
            pass

        # AUC 汇总
        auc_arr = np.asarray(auc_list, dtype=float)
        auc_mean = float(np.mean(auc_arr))
        auc_std = float(np.std(auc_arr)) if auc_arr.size > 1 else 0.0
        if ci_mode.lower() == 'percentile' and auc_arr.size > 1:
            alpha = (1.0 - float(ci_level)) / 2.0
            lo = float(np.percentile(auc_arr, 100*alpha))
            hi = float(np.percentile(auc_arr, 100*(1.0-alpha)))
            auc_text = f"AUC(mean)={auc_mean:.3f} | PI[{int(100*ci_level)}%]={lo:.3f}-{hi:.3f}"
        else:
            auc_text = f"AUC(mean)={auc_mean:.3f}±{auc_std:.3f}"

        # 标题追加 AUC 信息
        if auc_text:
            try:
                _t = plt.gca().get_title()
                if _t:
                    plt.title(f"{_t} | {auc_text}")
                else:
                    plt.title(auc_text)
            except Exception:
                pass
        plt.legend(loc='lower right')
        plt.tight_layout()

        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Repeat-mean ROC saved to {save_path}")
            except Exception as e:
                self.logger.warning(f"保存ROC失败: {e}")
        plt.show()

    def plot_within_disease_heatmap(self,
                                    matrix_csv: Union[str, Path],
                                    metric: str = 'auc',
                                    cmap: str = 'viridis',
                                    save_path: Optional[Union[str, Path]] = None,
                                    disease: Optional[str] = None,
                                    model: Optional[str] = None,
                                    order_csv: Optional[Union[str, Path]] = None,
                                    order_id_col: str = 'ProjectID',
                                    order_sort_col: str = 'ProjectOrder',
                                    colorbar_shrink: float = 0.8,
                                    colorbar_aspect: int = 25,
                                    fig_width_per_col: float = 0.6,
                                    fig_height_per_row: float = 0.6,
                                    fig_width_min: float = 8.0,
                                    fig_height_min: float = 6.0,
                                    fig_width_max: float = 18.0,
                                    fig_height_max: float = 12.0) -> None:
        """Plot heatmap for within_disease matrix CSV (train_proj x test_proj).

        Args:
            matrix_csv: Path to matrix CSV with columns ['train_proj','test_proj', <metric>]
            metric: Metric column to visualize ('auc' or 'accuracy')
            cmap: Matplotlib/Seaborn colormap name
            save_path: Optional path to save the figure
        """
        self.logger.info(f"Creating within_disease heatmap from {matrix_csv}...")
        matrix_csv = Path(matrix_csv)
        if not matrix_csv.exists():
            self.logger.warning(f"Matrix CSV not found: {matrix_csv}")
            return

        df = pd.read_csv(matrix_csv)
        metric_col = metric.lower()
        if metric_col not in df.columns:
            self.logger.warning(f"Metric column '{metric_col}' not in CSV; available: {list(df.columns)}")
            return

        # Pivot to wide matrix: rows=train_proj, cols=test_proj
        wide = df.pivot(index='train_proj', columns='test_proj', values=metric_col)
        # Optional: apply explicit order from CSV (e.g., between_project)
        if order_csv:
            try:
                ord_df = pd.read_csv(str(order_csv))
                # Require both id and sort columns; fallbacks if missing
                if order_sort_col in ord_df.columns:
                    ord_df = ord_df.sort_values(order_sort_col)
                if order_id_col in ord_df.columns:
                    order_list = [str(x) for x in ord_df[order_id_col].dropna().astype(str).tolist()]
                    # Intersect with existing labels
                    idx = [x for x in order_list if x in wide.index]
                    cols = [x for x in order_list if x in wide.columns]
                    if idx:
                        wide = wide.reindex(index=idx)
                    if cols:
                        wide = wide.reindex(columns=cols)
            except Exception as e:
                self.logger.warning(f"Failed to apply order from {order_csv}: {e}")
        # Try to infer disease/model from filename if not provided
        title_parts = []
        if disease is None or model is None:
            name = matrix_csv.stem  # e.g., within_disease_ASD_lasso_auc_matrix
            parts = name.split('_')
            # naive parse: ... within_disease <DISEASE> <MODEL> <METRIC> matrix
            if disease is None:
                for i, p in enumerate(parts):
                    if p.lower() == 'within' and i+2 < len(parts):
                        disease = parts[i+2]
                        break
                # fallback: look for 'ASD'/'AD' etc in parts
                if disease is None:
                    for p in parts:
                        if p.isupper() and len(p) <= 6:
                            disease = p; break
            if model is None:
                # heuristic: model sits between disease and metric
                for m in ['lasso','elasticnet','logistic','randomforest','catboost','neuralnetwork']:
                    if m in parts:
                        model = m; break
        if disease:
            title_parts.append(f"Disease={disease}")
        if model:
            title_parts.append(f"Model={model}")
        subtitle = " | ".join(title_parts)

        # Figure size with caps to avoid overly tall figures
        n_cols = len(wide.columns)
        n_rows = len(wide.index)
        width = max(fig_width_min, 1 + fig_width_per_col * n_cols)
        height = max(fig_height_min, 1 + fig_height_per_row * n_rows)
        width = min(width, fig_width_max)
        height = min(height, fig_height_max)
        plt.figure(figsize=(width, height))
        ax = sns.heatmap(
            wide,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'shrink': float(colorbar_shrink), 'aspect': int(colorbar_aspect)}
        )
        title_main = f"Within Disease Heatmap ({metric_col.upper()})"
        if subtitle:
            plt.title(f"{title_main}\n{subtitle}")
        else:
            plt.title(title_main)
        plt.xlabel('Test Project')
        plt.ylabel('Train Project')
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Heatmap saved to {save_path}")
        plt.show()

    def plot_within_disease_boxplot(self,
                                    matrices: List[Union[str, Path]],
                                    model_names: List[str],
                                    metric: str = 'auc',
                                    mode: str = 'self',  # 'self' (train==test) or 'cross' (train!=test)
                                    save_path: Optional[Union[str, Path]] = None,
                                    title: Optional[str] = None,
                                    show_points: bool = True,
                                    point_alpha: float = 0.6,
                                    point_size: float = 3.0,
                                    point_jitter: float = 0.18,
                                    model_order: Optional[Union[str, List[str]]] = 'median_desc',
                                    show_grid: bool = False,
                                    fig_width_per_model: float = 1.2,
                                    fig_width_min: float = 6.0,
                                    fig_width_max: float = 10.0,
                                    fig_height: float = 6.0) -> None:
        """Plot boxplot across models for within_disease matrices.

        Args:
            matrices: list of matrix CSV paths (each for one model)
            model_names: same-length list of model names to label x-axis
            metric: 'auc' or 'accuracy'
            mode: 'self' for diagonal (train==test), 'cross' for off-diagonal (train!=test)
            save_path: optional path to save figure
            title: optional title string
        """
        import pandas as _pd
        self.logger.info(f"Creating within_disease boxplot | mode={mode} | metric={metric}...")
        metric_col = metric.lower()

        rows = []
        for csv_path, model in zip(matrices, model_names):
            csv_path = Path(csv_path)
            if not csv_path.exists():
                self.logger.warning(f"Matrix missing, skip: {csv_path}")
                continue
            df = _pd.read_csv(csv_path)
            if metric_col not in df.columns:
                self.logger.warning(f"Metric '{metric_col}' missing in {csv_path.name}, skip")
                continue
            if mode == 'self':
                sub = df[df['train_proj'] == df['test_proj']]
            else:
                sub = df[df['train_proj'] != df['test_proj']]
            vals = sub[metric_col].dropna().values.tolist()
            for v in vals:
                rows.append({'model': model, metric_col: float(v)})

        if not rows:
            self.logger.warning("No data available for boxplot")
            return

        data = _pd.DataFrame(rows)

        # Determine x-order
        order = None
        if isinstance(model_order, list) and model_order:
            order = [m for m in model_order if m in data['model'].unique()]
        elif isinstance(model_order, str):
            if model_order.lower() == 'median_desc':
                med = data.groupby('model')[metric_col].median().sort_values(ascending=False)
                order = med.index.tolist()
            elif model_order.lower() == 'median_asc':
                med = data.groupby('model')[metric_col].median().sort_values(ascending=True)
                order = med.index.tolist()
            elif model_order.lower() == 'alphabetical':
                order = sorted(data['model'].unique())

        n_models = len((order or data['model'].unique()))
        fig_w = max(fig_width_min, fig_width_per_model * n_models)
        fig_w = min(fig_w, fig_width_max)
        plt.figure(figsize=(fig_w, fig_height))
        sns.boxplot(data=data, x='model', y=metric_col, order=order, palette='Set2')
        if show_points:
            # Overlay jittered points
            sns.stripplot(data=data, x='model', y=metric_col, order=order,
                          color='black', size=point_size, alpha=point_alpha, jitter=point_jitter)
        plt.ylim(0.0, 1.0)
        plt.ylabel(metric_col.upper())
        mode_lab = 'Self (Train=Test)' if mode == 'self' else 'Cross (Train→Test)'
        plt.title(title or f"Within Disease Boxplot | {mode_lab} | {metric_col.upper()}")
        # Toggle grid
        if not show_grid:
            plt.grid(False)
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Boxplot saved to {save_path}")
        plt.show()

    def plot_within_disease_boxplot_grouped(self,
                                            matrices: List[Union[str, Path]],
                                            model_names: List[str],
                                            metric: str = 'auc',
                                            save_path: Optional[Union[str, Path]] = None,
                                            title: Optional[str] = None) -> None:
        """Plot grouped boxplot (hue by Self/Cross) for the same metric across models.

        Args:
            matrices: list of matrix CSV paths (each for one model)
            model_names: same-length list of model names to label x-axis
            metric: 'auc' or 'accuracy'
            save_path: optional path to save figure
            title: optional title string
        """
        import pandas as _pd
        self.logger.info(f"Creating grouped within_disease boxplot | metric={metric}...")
        metric_col = metric.lower()

        rows = []
        for csv_path, model in zip(matrices, model_names):
            csv_path = Path(csv_path)
            if not csv_path.exists():
                self.logger.warning(f"Matrix missing, skip: {csv_path}")
                continue
            df = _pd.read_csv(csv_path)
            if metric_col not in df.columns:
                self.logger.warning(f"Metric '{metric_col}' missing in {csv_path.name}, skip")
                continue
            # Self (diagonal)
            sub_self = df[df['train_proj'] == df['test_proj']]
            for v in sub_self[metric_col].dropna().values.tolist():
                rows.append({'model': model, metric_col: float(v), 'mode': 'Self'})
            # Cross (off-diagonal)
            sub_cross = df[df['train_proj'] != df['test_proj']]
            for v in sub_cross[metric_col].dropna().values.tolist():
                rows.append({'model': model, metric_col: float(v), 'mode': 'Cross'})

        if not rows:
            self.logger.warning("No data available for grouped boxplot")
            return

        data = _pd.DataFrame(rows)
        plt.figure(figsize=(max(7, 2.5 * len(data['model'].unique())), 6.5))
        sns.boxplot(data=data, x='model', y=metric_col, hue='mode', palette='Set2')
        plt.ylim(0.0, 1.0)
        plt.ylabel(metric_col.upper())
        plt.xlabel('Model')
        plt.title(title or f"Within Disease Boxplot | {metric_col.upper()} | Self vs Cross")
        plt.legend(title='Mode', loc='best')
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Grouped boxplot saved to {save_path}")
        plt.show()

    def plot_disease_mode_boxplot(self,
                                   matrices: List[Union[str, Path]],
                                   metric: str,
                                   metadata_file: Union[str, Path],
                                   save_path: Optional[Union[str, Path]] = None,
                                   title: Optional[str] = None,
                                   show_points: bool = True,
                                   point_alpha: float = 0.6,
                                   point_size: float = 2.5,
                                   point_jitter: float = 0.15,
                                   fig_width: float = 10.0,
                                   fig_height: float = 6.0) -> None:
        """Plot boxplot with x=Disease, y=metric, hue=Mode(Self/Cross) using between_project matrices.

        Only counts pairs where train/test projects belong to the same disease.
        """
        import pandas as _pd
        metric_col = str(metric).lower()
        self.logger.info(f"Creating disease-mode boxplot | metric={metric_col}...")

        # Load metadata for mapping Project -> Disease
        md = _pd.read_csv(str(metadata_file), index_col=0)
        if 'Project' not in md.columns or 'Disease' not in md.columns:
            self.logger.warning("metadata需包含 'Project' 与 'Disease' 列，跳过绘图")
            return
        md_clean = md[['Project', 'Disease']].dropna()
        proj_to_dis = md_clean.drop_duplicates().set_index('Project')['Disease'].to_dict()
        # 统计每个疾病包含的项目数量
        dis_proj_counts = (md_clean.drop_duplicates()
                           .groupby('Disease')['Project']
                           .nunique()
                           .to_dict())

        rows = []
        for csv_path in matrices:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                continue
            try:
                df = _pd.read_csv(csv_path)
            except Exception:
                continue
            if not set(['train_proj', 'test_proj']).issubset(df.columns) or metric_col not in df.columns:
                continue
            for _, r in df.iterrows():
                tr = str(r['train_proj']); te = str(r['test_proj'])
                if tr not in proj_to_dis or te not in proj_to_dis:
                    continue
                dis_tr = proj_to_dis[tr]; dis_te = proj_to_dis[te]
                if dis_tr != dis_te:
                    continue  # only within same disease
                # 跳过仅有单一项目的疾病（无意义的Self/Cross划分）
                if dis_tr not in dis_proj_counts or dis_proj_counts[dis_tr] <= 1:
                    continue
                val = r.get(metric_col, None)
                if _pd.isna(val):
                    continue
                mode = 'Self' if tr == te else 'Cross'
                rows.append({'Disease': dis_tr, metric_col: float(val), 'Mode': mode})

        if not rows:
            self.logger.warning("No data available for disease-mode boxplot")
            return

        data = _pd.DataFrame(rows)
        plt.figure(figsize=(fig_width, fig_height))
        disease_order = sorted(data['Disease'].unique().tolist())
        hue_order = ['Self', 'Cross'] if set(data['Mode'].unique()) == set(['Self','Cross']) else None
        sns.boxplot(data=data, x='Disease', y=metric_col, hue='Mode', order=disease_order, hue_order=hue_order, palette='Set2')
        if show_points:
            sns.stripplot(data=data, x='Disease', y=metric_col, hue='Mode',
                          order=disease_order, hue_order=hue_order,
                          dodge=True, color='black', size=point_size,
                          alpha=point_alpha, jitter=point_jitter)
            # Avoid double legend entries from stripplot
            from matplotlib.patches import Patch
            handles, labels = plt.gca().get_legend_handles_labels()
            uniq = []
            seen = set()
            for h, l in zip(handles, labels):
                if l not in seen:
                    uniq.append((h, l)); seen.add(l)
            plt.legend([h for h, _ in uniq], [l for _, l in uniq], title='Mode', loc='best')
        plt.ylim(0.0, 1.0)
        plt.ylabel(metric_col.upper())
        plt.title(title or f"Within Disease Boxplot by Mode | {metric_col.upper()}")
        # Disable grid for cleaner boxplot
        plt.grid(False)
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Disease-mode boxplot saved to {save_path}")
        plt.show()
