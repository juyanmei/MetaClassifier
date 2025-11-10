"""
标准输出管理器。

基于原始metaClassifier项目的输出结构，为metaClassifier v1.0提供标准化的结果输出管理。
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_curve

from ..utils.logger import get_logger


class StandardOutputManager:
    """
    标准输出管理器。
    
    基于原始metaClassifier项目的输出结构，提供标准化的结果输出管理。
    """
    
    def __init__(self, base_output_dir: Union[str, Path] = "results"):
        """
        初始化标准输出管理器。
        
        Args:
            base_output_dir: 基础输出目录
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = get_logger("StandardOutputManager")
        # 保存最近一次根据预测概率计算得到的决策阈值，供写入final_run.yaml
        self.last_decision_threshold: Optional[float] = None
        
        # 定义标准子目录结构
        self.standard_subdirs = {
            'performance_metrics': '1_performance_metrics',
            'final_model': '2_final_model', 
            'feature_analysis': '3_feature_analysis',
            'reproducibility': '4_reproducibility'
        }
    
    def create_model_output_structure(self, 
                                    scope_name: str = "All",
                                    model_name: str = "lasso",
                                    data_type: str = "abundance_raw") -> Path:
        """
        创建模型输出目录结构。
        
        Args:
            scope_name: 范围名称 (All, project_name, disease_name)
            model_name: 模型名称
            data_type: 数据类型标签
            
        Returns:
            模型输出根目录路径
        """
        # 构建输出路径: results/{data_type}/builds/{scope_name}/{model_name}
        model_output_dir = self.base_output_dir / data_type / "builds" / scope_name / model_name
        
        # 创建所有标准子目录
        for subdir_name in self.standard_subdirs.values():
            (model_output_dir / subdir_name).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"创建模型输出结构: {model_output_dir}")
        return model_output_dir
    
    def create_report_output_structure(self, 
                                     report_type: str,
                                     scope_name: str = "All") -> Path:
        """
        创建报告输出目录结构。
        
        Args:
            report_type: 报告类型 (within_disease, between_project, between_disease, overall, models)
            scope_name: 范围名称
            
        Returns:
            报告输出目录路径
        """
        report_output_dir = self.base_output_dir / "reports" / report_type / scope_name
        report_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"创建报告输出结构: {report_output_dir}")
        return report_output_dir
    
    def save_performance_metrics(self, 
                               output_dir: Path,
                               nested_cv_results: Dict[str, Any],
                               model_name: str,
                               visualizer=None) -> None:
        """
        保存性能指标到标准位置。
        
        Args:
            output_dir: 模型输出目录
            nested_cv_results: 嵌套CV结果
            model_name: 模型名称
            visualizer: 可视化器实例（可选）
        """
        metrics_dir = output_dir / self.standard_subdirs['performance_metrics']
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存嵌套CV汇总结果（健壮性：仅当能匹配到模型时才写入）
        if 'nested_cv_results' in nested_cv_results:
            nested_results = nested_cv_results['nested_cv_results']
            if 'performance_metrics' in nested_results:
                performance_metrics = nested_results['performance_metrics']
                # 查找匹配的模型名称（不区分大小写）
                actual_model_name = None
                for key in performance_metrics.keys():
                    if key.lower() == model_name.lower():
                        actual_model_name = key
                        break
                if actual_model_name:
                    model_metrics = performance_metrics.get(actual_model_name, None)
                    if model_metrics:
                        # 转换为DataFrame格式
                        summary_data = []
                        # 收集所有指标名称（去掉_mean和_std后缀）
                        metric_names = set()
                        for metric_name in model_metrics.keys():
                            if metric_name.endswith('_mean') or metric_name.endswith('_std'):
                                base_name = metric_name.replace('_mean', '').replace('_std', '')
                                metric_names.add(base_name)
                        # 为每个指标创建一行数据
                        for metric_name in metric_names:
                            mean_value = model_metrics.get(f'{metric_name}_mean', 0.0)
                            std_value = model_metrics.get(f'{metric_name}_std', 0.0)
                            summary_data.append({
                                'model': model_name,
                                'metric': metric_name,
                                'mean': mean_value,
                                'std': std_value
                            })
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_csv(metrics_dir / 'nested_cv_summary.csv', index=False)
        
        # 2. 保存嵌套CV原始分数
        if 'nested_cv_results' in nested_cv_results and 'outer_fold_results' in nested_cv_results['nested_cv_results']:
            outer_fold_results = nested_cv_results['nested_cv_results']['outer_fold_results']
            raw_scores_data = []
            proba_rows = []
            
            for fold_idx, fold_result in enumerate(outer_fold_results):
                if 'model_results' in fold_result:
                    # 查找匹配的模型名称（不区分大小写）
                    actual_model_name = None
                    for key in fold_result['model_results'].keys():
                        if key.lower() == model_name.lower():
                            actual_model_name = key
                            break
                    
                    if actual_model_name:
                        model_result = fold_result['model_results'][actual_model_name]
                        if 'metrics' in model_result:
                            metrics = model_result['metrics']
                            row = {
                                'outer_fold': fold_result.get('fold_idx', fold_idx),
                                'repeat': fold_result.get('repeat_idx', 0),
                                'model_name': model_name
                            }
                            
                            # 添加测试队列信息（仅LOCO模式）
                            if 'test_cohort_name' in fold_result and fold_result['test_cohort_name'] is not None:
                                row['test_cohort'] = fold_result['test_cohort_name']
                            
                            row.update(metrics)
                            raw_scores_data.append(row)

                        # 生成样本级概率行
                        y_true_list = fold_result.get('y_test', [])
                        sample_ids = fold_result.get('test_sample_ids', [])
                        y_pred_list = model_result.get('predictions', [])
                        y_pred_proba = model_result.get('probabilities', [])
                        # 处理二分类概率展平
                        for i in range(min(len(sample_ids), len(y_true_list), len(y_pred_list), len(y_pred_proba))):
                            prob_entry = y_pred_proba[i]
                            if isinstance(prob_entry, (list, tuple)) and len(prob_entry) == 2:
                                prob_0, prob_1 = prob_entry[0], prob_entry[1]
                            else:
                                # 兼容性：若为标量或一维，尽力推断正类概率
                                prob_1 = prob_entry[1] if isinstance(prob_entry, (list, tuple)) and len(prob_entry) > 1 else (prob_entry if not isinstance(prob_entry, (list, tuple)) else None)
                                prob_0 = (1 - prob_1) if isinstance(prob_1, (int, float)) else None
                            proba_rows.append({
                                'sample_id': sample_ids[i],
                                'outer_fold': fold_result.get('fold_idx', fold_idx),
                                'repeat': fold_result.get('repeat_idx', 0),
                                'model_name': model_name,
                                'y_true': y_true_list[i],
                                'y_pred': y_pred_list[i],
                                'prob_0': prob_0,
                                'prob_1': prob_1
                            })
            
            if raw_scores_data:
                raw_scores_df = pd.DataFrame(raw_scores_data)
                raw_scores_df.to_csv(metrics_dir / 'nested_cv_raw_scores.csv', index=False)
            if proba_rows:
                proba_df = pd.DataFrame(proba_rows)
                proba_df.to_csv(metrics_dir / 'nested_cv_pred_proba.csv', index=False)

            # 基于样本级中位数聚合prob_1，计算Youden J阈值（仅记录阈值，不在此处输出HI文件）
            try:
                # 仅使用prob_1列，按sample_id聚合中位数；true_value取首次出现值
                grouped = (
                    proba_df.groupby('sample_id', as_index=False)
                    .agg(
                        true_value=('y_true', 'first'),
                        prob_1=('prob_1', 'median')
                    )
                )

                # 丢弃缺失prob_1的样本
                grouped = grouped.dropna(subset=['prob_1'])
                if not grouped.empty:
                    y_true = grouped['true_value'].to_numpy()
                    y_score = grouped['prob_1'].to_numpy(dtype=float)

                    # 计算ROC并根据Youden J选择最佳阈值
                    fpr, tpr, thresholds = roc_curve(y_true, y_score)
                    youden_j = tpr - fpr
                    best_idx = int(np.nanargmax(youden_j)) if youden_j.size > 0 else 0
                    best_thr = float(thresholds[best_idx]) if thresholds.size > 0 else 0.5

                    # 仅记录阈值，供写入final_run.yaml
                    self.last_decision_threshold = best_thr
                else:
                    self.logger.warning("nested_cv_pred_proba.csv 聚合后为空，跳过阈值计算")
            except Exception as e:
                self.logger.warning(f"计算阈值失败: {e}")
        
        # 3. 保存汇总指标JSON
        # 3. 保存汇总指标JSON（健壮性：找不到模型则写空dict）
        nested_perf = nested_cv_results.get('nested_cv_results', {}).get('performance_metrics', {})
        # 匹配大小写
        actual_model_name = None
        for key in nested_perf.keys():
            if isinstance(key, str) and key.lower() == model_name.lower():
                actual_model_name = key
                break
        matched_metrics = nested_perf.get(actual_model_name, {}) if actual_model_name else {}
        summary_metrics = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'nested_cv_results': matched_metrics,
            'consensus_features_count': len(nested_cv_results.get('consensus_features', {}).get(model_name, [])),
            'n_outer_folds': len(nested_cv_results.get('nested_cv_results', {}).get('outer_fold_results', []))
        }
        
        # 注释掉summary_metrics.json的生成，信息过于详细
        # with open(metrics_dir / 'summary_metrics.json', 'w', encoding='utf-8') as f:
        #     json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
        
        # 4. 保存性能指标图表（健壮性处理：找不到对应模型时跳过）
        if visualizer and isinstance(nested_cv_results, dict) and 'nested_cv_results' in nested_cv_results:
            try:
                nested_results = nested_cv_results.get('nested_cv_results', {})
                performance_metrics = nested_results.get('performance_metrics', {})
                # 查找匹配的模型名称（不区分大小写）
                actual_model_name = None
                for key in performance_metrics.keys():
                    if isinstance(key, str) and key.lower() == model_name.lower():
                        actual_model_name = key
                        break
                if actual_model_name:
                    model_metrics = performance_metrics.get(actual_model_name, None)
                    if model_metrics:
                        visualizer.plot_cv_metrics(
                            {'metrics': model_metrics},
                            save_path=metrics_dir / 'cv_metrics.png'
                        )
                        self.logger.info(f"CV指标图表已保存到: {metrics_dir / 'cv_metrics.png'}")
                    else:
                        self.logger.warning("未找到对应模型的性能指标，跳过CV指标图表绘制")
                else:
                    self.logger.warning("性能指标中未匹配到当前模型名称，跳过CV指标图表绘制")
            except Exception as e:
                self.logger.warning(f"保存CV指标图表失败: {e}")
        
        self.logger.info(f"性能指标已保存到: {metrics_dir}")
    
    def save_final_model(self, 
                        output_dir: Path,
                        final_model: Any,
                        model_name: str) -> None:
        """
        保存最终模型到标准位置。
        
        Args:
            output_dir: 模型输出目录
            final_model: 最终模型对象
            model_name: 模型名称
        """
        final_model_dir = output_dir / self.standard_subdirs['final_model']
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        import joblib
        joblib.dump(final_model, final_model_dir / 'final_model.joblib')
        
        # 保存模型信息
        model_info = {
            'model_name': model_name,
            'model_type': type(final_model).__name__,
            'timestamp': datetime.now().isoformat(),
            'parameters': getattr(final_model, 'get_params', lambda: {})()
        }
        
        # 注释掉model_info.json的生成，信息较少且冗余
        # with open(final_model_dir / 'model_info.json', 'w', encoding='utf-8') as f:
        #     json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"最终模型已保存到: {final_model_dir}")
    
    def save_feature_analysis(self, 
                            output_dir: Path,
                            consensus_features: Dict[str, list],
                            feature_importance: Optional[Dict[str, Any]] = None,
                            model_name: str = None,
                            nested_cv_results: Optional[Dict[str, Any]] = None,
                            all_features: Optional[List[str]] = None,
                            constant_removed: Optional[set] = None,
                            variance_removed: Optional[set] = None) -> None:
        """
        保存特征分析结果到标准位置。
        
        Args:
            output_dir: 模型输出目录
            consensus_features: 共识特征
            feature_importance: 特征重要性
            model_name: 模型名称
            nested_cv_results: 嵌套CV结果
            all_features: 所有特征列表
            constant_removed: 被常量过滤移除的特征
            variance_removed: 被方差过滤移除的特征
        """
        feature_analysis_dir = output_dir / self.standard_subdirs['feature_analysis']
        feature_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存共识特征（保持原有格式不变）
        if model_name and consensus_features:
            # 查找匹配的模型名称（不区分大小写）
            actual_model_name = None
            for key in consensus_features.keys():
                if key.lower() == model_name.lower():
                    actual_model_name = key
                    break
            
            if actual_model_name:
                features = consensus_features[actual_model_name]
                # 计算repeat频率
                repeat_freq_map = {}
                repeats = set()
                if isinstance(nested_cv_results, dict):
                    nested = nested_cv_results.get('nested_cv_results', {})
                    outer_fold_results = nested.get('outer_fold_results', [])
                    # 计算repeat频率：统计每个特征在多少个repeat中出现
                    repeat_to_features = {}  # {repeat_idx: {feature: count_in_this_repeat}}
                    for fold in outer_fold_results:
                        fold_features = []
                        mr = fold.get('model_results', {})
                        # 模型名大小写匹配
                        match_model = None
                        for k in mr.keys():
                            if isinstance(k, str) and k.lower() == model_name.lower():
                                match_model = k
                                break
                        if match_model is None:
                            continue
                        fold_features = mr[match_model].get('selected_features', [])
                        # repeat计数
                        ridx = fold.get('repeat_idx', 0)
                        repeats.add(ridx)
                        # 初始化该repeat的特征计数
                        if ridx not in repeat_to_features:
                            repeat_to_features[ridx] = {}
                        # 统计该特征在该repeat中的出现次数
                        for f in fold_features:
                            repeat_to_features[ridx][f] = repeat_to_features[ridx].get(f, 0) + 1
                    
                    num_repeats = len(repeats) if repeats else 0
                    if num_repeats > 0:
                        # 计算repeat频率：特征在多少个repeat中出现（至少出现1次就算出现）
                        for ridx, feature_counts in repeat_to_features.items():
                            for f in feature_counts.keys():
                                repeat_freq_map[f] = repeat_freq_map.get(f, 0) + 1
                # 组装数据框
                rows = []
                for i, f in enumerate(features, start=1):
                    repeat_freq = (repeat_freq_map.get(f, 0) / len(repeats)) if repeats else None
                    rows.append({
                        'feature_name': f,
                        'rank': i,
                        'repeat_selection_ratio': repeat_freq
                    })
                features_df = pd.DataFrame(rows)
                features_df.to_csv(feature_analysis_dir / 'consensus_features.csv', index=False)

        # 2. 生成特征处理追踪文件（方案B）
        try:
            if all_features and isinstance(all_features, (list, tuple)):
                # 计算repeat频率
                repeat_freq_map = {}
                repeats = set()
                if isinstance(nested_cv_results, dict):
                    nested = nested_cv_results.get('nested_cv_results', {})
                    outer_fold_results = nested.get('outer_fold_results', [])
                    # repeat频率：出现过即计一次repeat
                    repeat_to_features = {}
                    for fold in outer_fold_results:
                        mr = fold.get('model_results', {})
                        match_model = None
                        for k in mr.keys():
                            if isinstance(k, str) and k.lower() == (model_name or '').lower():
                                match_model = k
                                break
                        if match_model is None:
                            continue
                        fold_features = mr[match_model].get('selected_features', [])
                        ridx = fold.get('repeat_idx', 0)
                        repeats.add(ridx)
                        if ridx not in repeat_to_features:
                            repeat_to_features[ridx] = set()
                        for f in fold_features:
                            repeat_to_features[ridx].add(f)
                    for ridx, fset in repeat_to_features.items():
                        for f in fset:
                            repeat_freq_map[f] = repeat_freq_map.get(f, 0) + 1
                # 归一化基数
                denom_repeat = float(len(repeats)) if repeats else None

                # 组装特征处理追踪表（方案B）
                rows_trace = []
                const_set = set(constant_removed or [])
                var_set = set(variance_removed or [])
                for f in all_features:
                    # 确定预处理状态
                    if f in const_set:
                        preprocessing_status = "constant_removed"
                    elif f in var_set:
                        preprocessing_status = "variance_filtered"
                    else:
                        preprocessing_status = "passed"
                    
                    # 计算选择比例
                    selection_ratio = (repeat_freq_map.get(f, 0) / denom_repeat) if denom_repeat else 0.0
                    
                    # 确定最终状态
                    if preprocessing_status != "passed":
                        final_status = "excluded"
                    elif selection_ratio == 0.0:
                        final_status = "excluded"
                    elif selection_ratio == 1.0:
                        final_status = "consensus_selected"
                    else:
                        final_status = "partially_selected"
                    
                    rows_trace.append({
                        'feature_name': f,
                        'preprocessing_status': preprocessing_status,
                        'selection_ratio': selection_ratio,
                        'final_status': final_status
                    })
                df_trace = pd.DataFrame(rows_trace)
                df_trace.to_csv(feature_analysis_dir / 'feature_processing_trace.csv', index=False)
        except Exception as e:
            self.logger.warning(f"保存特征处理追踪文件失败: {e}")

        # 3. 生成详细特征选择文件
        try:
            if all_features and isinstance(all_features, (list, tuple)) and isinstance(nested_cv_results, dict):
                nested = nested_cv_results.get('nested_cv_results', {})
                outer_fold_results = nested.get('outer_fold_results', [])
                
                # 收集每个特征在每个repeat和每个outer fold中的选择情况
                feature_selection_details = {}  # {feature: {repeat_outer: ratio}}
                repeats = set()
                outer_folds = set()
                
                for fold in outer_fold_results:
                    mr = fold.get('model_results', {})
                    match_model = None
                    for k in mr.keys():
                        if isinstance(k, str) and k.lower() == (model_name or '').lower():
                            match_model = k
                            break
                    if match_model is None:
                        continue
                    
                    fold_features = mr[match_model].get('selected_features', [])
                    ridx = fold.get('repeat_idx', 0)
                    fold_idx = fold.get('fold_idx', 0)
                    repeats.add(ridx)
                    outer_folds.add(fold_idx)
                    
                    # 记录该特征在该repeat和outer fold中被选中
                    for f in fold_features:
                        if f not in feature_selection_details:
                            feature_selection_details[f] = {}
                        key = f"repeat{ridx}_outerCV{fold_idx}"
                        feature_selection_details[f][key] = 1.0
                
                # 只为通过预处理的特征创建详细记录
                rows_detailed = []
                const_set = set(constant_removed or [])
                var_set = set(variance_removed or [])
                
                for f in all_features:
                    # 只处理通过预处理的特征（既不是常量移除也不是方差过滤的特征）
                    if f not in const_set and f not in var_set:
                        row = {'feature_name': f}
                        
                        # 为每个repeat和outer fold组合创建列
                        for ridx in sorted(repeats):
                            for fold_idx in sorted(outer_folds):
                                key = f"repeat{ridx}_outerCV{fold_idx}"
                                ratio = feature_selection_details.get(f, {}).get(key, 0.0)
                                row[key] = ratio
                        
                        # 计算总体比例
                        total_selections = sum(feature_selection_details.get(f, {}).values())
                        total_possibilities = len(repeats) * len(outer_folds)
                        overall_ratio = total_selections / total_possibilities if total_possibilities > 0 else 0.0
                        row['overall_ratio'] = overall_ratio
                        
                        rows_detailed.append(row)
                
                df_detailed = pd.DataFrame(rows_detailed)
                df_detailed.to_csv(feature_analysis_dir / 'feature_selection_detailed.csv', index=False)
        except Exception as e:
            self.logger.warning(f"保存详细特征选择文件失败: {e}")
        
        # 4. 保存特征重要性
        if feature_importance and model_name and model_name in feature_importance:
            importance_data = feature_importance[model_name]
            if isinstance(importance_data, dict):
                importance_df = pd.DataFrame(list(importance_data.items()), 
                                          columns=['feature_name', 'importance'])
                importance_df = importance_df.sort_values('importance', ascending=False)
                importance_df.to_csv(feature_analysis_dir / 'feature_importance.csv', index=False)
        
        self.logger.info(f"特征分析结果已保存到: {feature_analysis_dir}")
    
    def save_reproducibility_info(self, 
                                output_dir: Path,
                                config: Dict[str, Any],
                                model_name: str) -> None:
        """
        保存可重现性信息到标准位置。
        
        Args:
            output_dir: 模型输出目录
            config: 配置信息
            model_name: 模型名称
        """
        reproducibility_dir = output_dir / self.standard_subdirs['reproducibility']
        reproducibility_dir.mkdir(parents=True, exist_ok=True)
        
        # 若已存在final_run.yaml，则合并已有内容（保留既有字段）
        import yaml
        existing_info = None
        final_run_path = reproducibility_dir / 'final_run.yaml'
        if final_run_path.exists():
            try:
                with open(final_run_path, 'r', encoding='utf-8') as rf:
                    existing_info = yaml.safe_load(rf) or {}
            except Exception:
                existing_info = None

        # 合并配置：在传入config基础上补充决策阈值与最终阶段参数；若之前已有且当前未提供，则保留之前的值
        merged_config = dict(config or {})
        if self.last_decision_threshold is not None:
            merged_config['decision_threshold'] = float(self.last_decision_threshold)
        else:
            # 保留已有
            if isinstance(existing_info, dict):
                prev_cfg = existing_info.get('config', {})
                if isinstance(prev_cfg, dict) and 'decision_threshold' in prev_cfg and 'decision_threshold' not in merged_config:
                    merged_config['decision_threshold'] = prev_cfg['decision_threshold']

        # 若调用方传入了最终阶段参数（如 final_cv_folds 等），则记录
        # 这里仅对常见键做规范化保留
        for k in ['final_cv_folds', 'final_search_method', 'final_cpu']:
            if k in merged_config:
                continue
            # 尝试从外层对象读取（若调用端约定传入）
            try:
                if hasattr(self, k):
                    merged_config[k] = getattr(self, k)
            except Exception:
                pass

        # 组装最终内容（尽量保留已有顶层字段）
        final_run_info = existing_info if isinstance(existing_info, dict) else {}
        final_run_info.update({
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'config': merged_config,
            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            'metaClassifier_version': '1.0.0'
        })
        
        # 将可能的 tuple 转为 list，避免后续安全加载 YAML 失败
        def _tuple_to_list(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _tuple_to_list(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_tuple_to_list(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj

        final_run_info_serializable = _tuple_to_list(final_run_info)
        with open(final_run_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(final_run_info_serializable, f, default_flow_style=False, allow_unicode=True)
        
        # 保存运行日志到4_reproducibility目录
        self._save_runtime_logs(reproducibility_dir)
        
        self.logger.info(f"可重现性信息已保存到: {reproducibility_dir}")
    
    def _save_runtime_logs(self, reproducibility_dir: Path):
        """保存运行时的日志信息到4_reproducibility目录"""
        # 所有日志记录由main.py的TeeLogger负责
        # 这个方法不再生成日志文件，避免覆盖TeeLogger的输出
        log_file = reproducibility_dir / 'run.log'
        self.logger.info(f"运行日志将保存到: {log_file}")
    
    def save_report_visualizations(self, 
                                 report_output_dir: Path,
                                 performance_metrics: Dict[str, Any],
                                 model_name: str,
                                 visualizer=None) -> None:
        """
        保存报告中的可视化图表。
        
        Args:
            report_output_dir: 报告输出目录
            performance_metrics: 性能指标
            model_name: 模型名称
            visualizer: 可视化器实例
        """
        if not visualizer or model_name not in performance_metrics:
            return
        
        model_metrics = performance_metrics[model_name]
        
        try:
            # 保存性能指标图表
            visualizer.plot_cv_metrics(
                {'metrics': model_metrics},
                save_path=report_output_dir / f'performance_metrics_{model_name}.png'
            )
            self.logger.info(f"报告图表已保存到: {report_output_dir / f'performance_metrics_{model_name}.png'}")
        except Exception as e:
            self.logger.warning(f"保存报告图表失败: {e}")
    
    def create_inspection_output(self, 
                               model_name: str,
                               test_data_source: str = "unknown") -> Path:
        """
        创建检查输出目录。
        
        Args:
            model_name: 模型名称
            test_data_source: 测试数据源
            
        Returns:
            检查输出目录路径
        """
        inspection_dir = self.base_output_dir / ".cache" / "inspections" / f"{model_name}_on_{test_data_source}"
        inspection_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建性能指标子目录
        (inspection_dir / self.standard_subdirs['performance_metrics']).mkdir(exist_ok=True)
        
        self.logger.info(f"检查输出目录已创建: {inspection_dir}")
        return inspection_dir
    
    def get_standard_output_path(self, 
                               output_type: str,
                               scope_name: str = "All",
                               model_name: str = "lasso",
                               data_type: str = "abundance_raw") -> Path:
        """
        获取标准输出路径。
        
        Args:
            output_type: 输出类型 (model, report, inspection)
            scope_name: 范围名称
            model_name: 模型名称
            data_type: 数据类型
            
        Returns:
            标准输出路径
        """
        if output_type == "model":
            return self.base_output_dir / data_type / "builds" / scope_name / model_name
        elif output_type == "report":
            return self.base_output_dir / "reports" / scope_name
        elif output_type == "inspection":
            return self.base_output_dir / ".cache" / "inspections" / f"{model_name}_inspection"
        else:
            raise ValueError(f"Unknown output type: {output_type}")
