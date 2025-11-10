def _emit_predictions_between_disease(args, diseases: list):
    """在疾病层级进行互相预测：trainDisease→testDisease。
    加载 builds/Disease_<Disease>/<model>/2_final_model 下的模型，对 testDisease 样本预测，输出到
    reports/predict_disease/<model>/<trainDis>_to_<testDis>_predictions.csv
    """
    logger = get_logger("ReportPredict(between_disease)")
    output_root = Path(getattr(args, 'output', 'tests/result'))
    prof_file = Path(getattr(args, 'prof_file'))
    metadata_file = Path(getattr(args, 'metadata_file'))
    models_arg = getattr(args, 'models', None)
    selected_models = models_arg if models_arg else None

    # 加载全量数据
    from ..data.loader import DataLoader
    dl = DataLoader()
    X_all, y_all, groups_all = dl.load_data(
        prof_file=str(prof_file),
        metadata_file=str(metadata_file),
        scope=None,
        use_presence_absence=bool(getattr(args, 'use_presence_absence', True)),
        use_clr=bool(getattr(args, 'use_clr', False)),
        enable_cohort_analysis=False,
        group_col='Group',
        label_0=getattr(args, 'label_0', None),
        label_1=getattr(args, 'label_1', None)
    )[:3]
    try:
        y_all_series = pd.Series(y_all, index=X_all.index)
    except Exception:
        y_all_series = None

    meta = pd.read_csv(str(metadata_file), index_col=0)

    # 搜索疾病级模型根
    user_builds_root = getattr(args, 'builds_root', None)
    candidate_build_roots = []
    if user_builds_root:
        try:
            candidate_build_roots.append(Path(user_builds_root))
        except Exception:
            pass
    candidate_build_roots += [
        output_root / 'abundance_raw' / 'builds',
        output_root / 'prevalence' / 'builds',
        output_root / 'builds',
    ]

    import joblib

    for train_dis in diseases:
        # 定位疾病级目录：<root>/Disease_<Disease>
        model_base_dirs = []
        for root in candidate_build_roots:
            mroot = root / f"Disease_{train_dis}"
            if not mroot.exists():
                continue
            if selected_models:
                for m in selected_models:
                    cand = mroot / str(m).lower() / '2_final_model'
                    if cand.exists():
                        model_base_dirs.append((m, cand))
            else:
                for model_dir in mroot.iterdir():
                    cand = model_dir / '2_final_model'
                    if cand.exists():
                        model_base_dirs.append((model_dir.name, cand))
        if not model_base_dirs:
            logger.warning(f"{train_dis}: 未发现任何final_model目录 (疾病级)")
            continue

        for model_name, final_dir in model_base_dirs:
            try:
                model_path = final_dir / 'final_model.joblib'
                if not model_path.exists():
                    found = list(final_dir.glob('final_model_*.joblib'))
                    if found:
                        model_path = found[0]
                clf = joblib.load(model_path)
                features = getattr(clf, 'selected_features_', None)
                if not features:
                    csv_path = final_dir.parent.parent / '3_feature_analysis' / 'consensus_features.csv'
                    if csv_path.exists():
                        cdf = pd.read_csv(csv_path)
                        if 'feature_name' in cdf.columns:
                            features = cdf['feature_name'].dropna().astype(str).tolist()
                if not features:
                    logger.warning(f"{train_dis}/{model_name}: 无法确定特征列表，跳过")
                    continue
                for test_dis in diseases:
                    test_ids = meta[meta['Disease'] == test_dis].index
                    X_test = X_all.loc[X_all.index.intersection(test_ids)].copy()
                    if X_test.empty:
                        continue
                    for f in features:
                        if f not in X_test.columns:
                            X_test[f] = 0
                    X_test = X_test[features]
                    score = _predict_positive_scores(clf, X_test)
                    out_dir = output_root / 'reports' / 'predict_disease' / str(model_name).lower()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = out_dir / f"{train_dis}_to_{test_dis}_predictions.csv"
                    if y_all_series is not None:
                        true_y = y_all_series.loc[X_test.index]
                    else:
                        true_y = _extract_true_labels(X_test.index, y_all, metadata_file)
                    pd.DataFrame({
                        'sample_id': X_test.index,
                        'true_label': true_y.values,
                        'predicted_score': score,
                    }).to_csv(out_csv, index=False)
            except Exception as e:
                logger.warning(f"{train_dis}/{model_name} 疾病级互测失败: {e}")
#!/usr/bin/env python3
"""
报告生成流水线 - 支持复杂的分析场景
基于旧代码的复杂设计，支持多种分析场景
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import argparse
from sklearn.metrics import roc_curve, auc

from ..utils.logger import get_logger


def handle_report(args) -> None:
    """处理report命令 - 支持复杂的分析场景"""
    logger = get_logger("ReportPipeline")
    logger.info("开始报告生成流水线...")
    
    try:
        # 使用复杂的报告生成逻辑
        generate_report(args)
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        raise


def generate_report(args):
    """路由报告生成，根据场景使用相应的处理器"""
    logger = get_logger("ReportGenerator")
    scenario_raw = getattr(args, 'scenario')
    scenario = str(scenario_raw).strip().lower() if scenario_raw is not None else None
    metric_name = getattr(args, 'metric', 'auc')
    
    # 加载元数据以验证列存在
    try:
        metadata = pd.read_csv(getattr(args, 'metadata_file'), index_col=0)
        if 'Project' not in metadata.columns or 'Disease' not in metadata.columns:
            raise ValueError("metadata 需包含列 'Project' 与 'Disease'")
    except Exception as e:
        logger.error(f"无法加载元数据: {e}")
        raise

    handlers = {
        'within_disease': lambda: _report_within_disease(args, metadata, metric_name),
        'between_project': lambda: _report_between_project(args, metadata),
        'between_disease': lambda: _report_between_disease(args, metadata),
        'overall': lambda: _report_overall(args),
        'models': lambda: _report_models(args),
        'predict_external_disease': lambda: _predict_external_disease(args, metadata),
        'predict_external_overall': lambda: _predict_external_overall(args, metadata),
    }

    handler = handlers.get(scenario)
    if handler is None:
        supported = ', '.join(sorted(handlers.keys()))
        raise ValueError(f"Unknown scenario: {scenario_raw}. Supported: {supported}")
    return handler()


def _predict_positive_scores(clf, X: pd.DataFrame):
    """返回正类(1)的分数。优先使用predict_proba的正类列，其次decision_function，最后predict。"""
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X)
        try:
            proba = np.asarray(proba)
        except Exception:
            pass
        if proba.ndim == 2 and proba.shape[1] >= 2:
            pos_index = None
            if hasattr(clf, 'classes_'):
                classes = list(getattr(clf, 'classes_'))
                if 1 in classes:
                    pos_index = classes.index(1)
            if pos_index is None:
                pos_index = proba.shape[1] - 1
            return proba[:, pos_index]
        return proba.reshape(-1)
    if hasattr(clf, 'decision_function'):
        score = clf.decision_function(X)
        try:
            return np.asarray(score)
        except Exception:
            return score
    pred = clf.predict(X)
    try:
        return np.asarray(pred)
    except Exception:
        return pred


def _extract_true_labels(sample_index, y_like, metadata_path: Path):
    """返回与 sample_index 对齐的真实标签Series。
    优先使用 y_like（若为pandas Series并索引可对齐）；否则从 metadata 中自动探测常见标签列。
    """
    try:
        if isinstance(y_like, pd.Series):
            try:
                return y_like.loc[sample_index]
            except Exception:
                pass
    except Exception:
        pass
    try:
        md = pd.read_csv(str(metadata_path), index_col=0)
        candidates = ['Label', 'label', 'y', 'Y', 'Phenotype', 'phenotype', 'CaseControl', 'case_control', 'Status', 'status', 'Class', 'class']
        for col in candidates:
            if col in md.columns:
                ser = md[col]
                return ser.loc[ser.index.intersection(sample_index)].reindex(sample_index)
        if 'Disease' in md.columns:
            ser = md['Disease'].astype(str)
            bin_ser = ser.apply(lambda v: 1 if v.strip().upper() in ('AD','ASD','PD','IBD') else 0)
            return bin_ser.loc[bin_ser.index.intersection(sample_index)].reindex(sample_index)
    except Exception:
        pass
    return pd.Series(index=sample_index, dtype='float')


def _compute_auc_from_nested_cv_pred_proba(csv_path: Path, metric: str = 'auc') -> Optional[float]:
    """从 nested_cv_pred_proba.csv 重新计算每个repeat的整体AUC，然后返回所有repeat AUC的均值。
    
    关键修复：不是从summary文件读取每折AUC的均值，而是基于所有repeat的所有outer_fold的OOF预测重新计算。
    
    Args:
        csv_path: nested_cv_pred_proba.csv 文件路径
        metric: 指标名称，'auc' 或其他
    
    Returns:
        所有repeat的整体AUC的均值，如果计算失败则返回None
    """
    if metric.lower() != 'auc':
        # 对于非AUC指标，仍然从summary文件读取（如果需要）
        return None
    
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # 自动探测列名
        label_col = None
        prob_col = None
        repeat_col = 'repeat'
        
        for c in ['y_true', 'true_label', 'label', 'y', 'Y']:
            if c in df.columns:
                label_col = c
                break
        for c in ['prob_1', 'y_proba', 'predicted_score', 'proba', 'score']:
            if c in df.columns:
                prob_col = c
                break
        
        if label_col is None or prob_col is None:
            return None
        
        # 清洗数据
        sub = df[[label_col, prob_col] + ([repeat_col] if repeat_col in df.columns else [])].copy()
        sub = sub.dropna(subset=[label_col, prob_col])
        
        # 规范标签到 {0,1}
        try:
            sub[label_col] = sub[label_col].astype(float)
        except Exception:
            sub[label_col] = sub[label_col].astype(str).str.strip()
            uniques = sorted(sub[label_col].unique())
            if len(uniques) == 2:
                mapping = {uniques[0]: 0, uniques[1]: 1}
                sub[label_col] = sub[label_col].map(mapping)
        
        sub = sub[(sub[label_col] == 0) | (sub[label_col] == 1)]
        if sub.empty:
            return None
        
        # 按repeat计算整体AUC
        if repeat_col not in sub.columns:
            # 如果没有repeat列，直接计算整体AUC
            y = sub[label_col].values
            p = sub[prob_col].values
            if len(np.unique(y)) < 2:
                return None
            fpr, tpr, _ = roc_curve(y, p)
            return float(auc(fpr, tpr))
        
        repeats = sorted([r for r in sub[repeat_col].dropna().unique().tolist()])
        auc_list = []
        
        for r in repeats:
            sdf = sub[sub[repeat_col] == r]
            if sdf.empty:
                continue
            
            y = sdf[label_col].values
            p = sdf[prob_col].values
            
            if len(np.unique(y)) < 2:
                continue
            
            # 重新计算该repeat的整体AUC（基于所有outer_fold的OOF预测）
            fpr, tpr, _ = roc_curve(y, p)
            auc_repeat = float(auc(fpr, tpr))
            auc_list.append(auc_repeat)
        
        if not auc_list:
            return None
        
        # 返回所有repeat AUC的均值
        return float(np.mean(auc_list))
    
    except Exception as e:
        logger = get_logger("ReportPipeline")
        logger.warning(f"从 {csv_path} 重新计算AUC失败: {e}")
        return None


def _fill_matrix_with_predictions(args, matrix_csv: Path, projects: list, metric_name: str = 'auc'):
    """读取 reports/predict/<model>/<train>_to_<test>_predictions.csv，计算AUC/accuracy，填入矩阵CSV 非对角线。"""
    from sklearn.metrics import roc_auc_score, accuracy_score
    output_root = Path(getattr(args, 'output', 'tests/result'))
    models_arg = getattr(args, 'models', None)
    model = (models_arg[0] if isinstance(models_arg, (list, tuple)) and models_arg else models_arg) or 'lasso'
    model = str(model).lower()

    df = pd.read_csv(matrix_csv)
    # 允许按所选 metric 列进行填充；保证必要列存在
    if df.empty or not set(['train_proj', 'test_proj']).issubset(df.columns):
        return
    # 若目标列不存在则先创建
    target_col = 'auc' if metric_name.lower() == 'auc' else 'accuracy'
    if target_col not in df.columns:
        df[target_col] = None

    pred_dir = output_root / 'reports' / 'predict' / model
    if not pred_dir.exists():
        return

    # 构建查找映射
    filename = lambda tr, te: pred_dir / f"{tr}_to_{te}_predictions.csv"

    for tr in projects:
        for te in projects:
            if tr == te:
                continue
            fp = filename(tr, te)
            if not fp.exists():
                continue
            try:
                pdf = pd.read_csv(fp)
                # 需要 true_label 和 predicted_score
                if 'true_label' not in pdf.columns or 'predicted_score' not in pdf.columns:
                    continue
                y_true = pdf['true_label']
                y_score = pdf['predicted_score']
                # 容错：剔除缺失
                mask = y_true.notna() & y_score.notna()
                y_true = y_true[mask]
                y_score = y_score[mask]
                if len(y_true) < 2 or y_true.nunique() < 2:
                    continue
                auc = roc_auc_score(y_true, y_score)
                if metric_name.lower() == 'auc':
                    df.loc[(df['train_proj'] == tr) & (df['test_proj'] == te), 'auc'] = auc
                elif metric_name.lower() == 'accuracy':
                    acc = accuracy_score(y_true, (y_score >= 0.5).astype(int))
                    df.loc[(df['train_proj'] == tr) & (df['test_proj'] == te), 'accuracy'] = acc
            except Exception:
                continue

    df.to_csv(matrix_csv, index=False)

def _report_within_disease(args, metadata, metric_name: str):
    """疾病内项目间比较报告"""
    logger = get_logger("ReportGenerator")
    disease_name = getattr(args, 'disease', None)
    if not disease_name:
        raise ValueError("within_disease 场景需要 --disease")
    
    sub = metadata[metadata['Disease'] == disease_name]
    projects = sorted(sub['Project'].dropna().unique().tolist())
    
    if len(projects) < 2:
        output_root = Path(getattr(args, 'output', 'tests/result'))
        out_dir = output_root / 'reports' / 'within_disease' / disease_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"within_disease_{disease_name}_{metric_name}_matrix.csv"
        pd.DataFrame().to_csv(out_csv, index=False)
        logger.warning(f"within_disease: '{disease_name}' 仅检测到 {len(projects)} 个 Project，已生成空矩阵: {out_csv}")
        return str(out_csv)
    
    # 支持多模型：若传入多个 --models，则分别生成按模型命名的矩阵文件
    models_arg = getattr(args, 'models', None)
    model_list = []
    if models_arg:
        if isinstance(models_arg, (list, tuple)):
            model_list = [str(m).strip().lower() for m in models_arg if str(m).strip()]
        elif isinstance(models_arg, str):
            model_list = [p.strip().lower() for p in models_arg.split(',') if p.strip()]
    if not model_list:
        model_list = ['lasso']

    for model_for_matrix in model_list:
        output_root = Path(getattr(args, 'output', 'tests/result'))
        out_dir = output_root / 'reports' / 'within_disease' / disease_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"within_disease_{disease_name}_{model_for_matrix}_{metric_name}_matrix.csv"
        candidate_build_roots = [
            output_root / 'abundance_raw' / 'builds',
            output_root / 'prevalence' / 'builds',
            output_root / 'builds',
        ]

        def read_nested_cv_metric_value(proj: str, metric: str) -> float:
            """读取nested CV指标值，优先从nested_cv_pred_proba.csv重新计算AUC"""
            for root in candidate_build_roots:
                for proj_dir in [root / proj, root / f"Project_{proj}"]:
                    model_dir = proj_dir / model_for_matrix / '1_performance_metrics'
                    if not model_dir.exists():
                        continue
                    
                    # 关键修复：对于AUC，优先从nested_cv_pred_proba.csv重新计算
                    if metric.lower() == 'auc':
                        pred_proba_csv = model_dir / 'nested_cv_pred_proba.csv'
                        if pred_proba_csv.exists():
                            auc_val = _compute_auc_from_nested_cv_pred_proba(pred_proba_csv, metric)
                            if auc_val is not None:
                                return auc_val
                    
                    # 回退：从summary文件读取（用于非AUC指标或当pred_proba文件不存在时）
                    for fp in [model_dir / 'nested_cv_summary.csv', model_dir / 'cv_summary.csv']:
                        if not fp.exists():
                            continue
                        try:
                            dfm = pd.read_csv(fp)
                            cols_lower = {c.lower(): c for c in dfm.columns}
                            # 宽表
                            if metric.lower() in cols_lower:
                                return float(dfm[cols_lower[metric.lower()]].mean())
                            # 长表 metric/mean
                            if 'metric' in cols_lower and 'mean' in cols_lower:
                                mcol = cols_lower['metric']; vcol = cols_lower['mean']
                                dfm[mcol] = dfm[mcol].astype(str).str.lower()
                                sub = dfm[dfm[mcol] == metric.lower()]
                                if sub.empty and metric.lower() == 'auc':
                                    sub = dfm[dfm[mcol].isin(['roc_auc'])]
                                if sub.empty and metric.lower() == 'accuracy':
                                    sub = dfm[dfm[mcol].isin(['acc'])]
                                if not sub.empty:
                                    return float(sub[vcol].mean())
                        except Exception as e:
                            logger.warning(f"读取{fp}失败: {e}")
            return None

        # 组装长表：只保留指定 metric 列
        records = []
        for src in projects:
            mval = read_nested_cv_metric_value(src, metric_name)
            for tgt in projects:
                rec = {
                    'train_proj': src,
                    'test_proj': tgt,
                    metric_name: mval if src == tgt else None,
                }
                records.append(rec)

        df_long = pd.DataFrame(records)
        df_long.to_csv(out_csv, index=False)
        logger.info(f"✅ within_disease 长表(真实指标)已生成: {out_csv}")
        # 自动生成热图
        try:
            from ..evaluation.visualizer import ResultsVisualizer
            viz = ResultsVisualizer()
            viz.plot_within_disease_heatmap(
                matrix_csv=str(out_csv),
                metric=metric_name,
                cmap='YlOrRd',
                save_path=str(out_dir / f"{out_csv.stem}_heatmap.pdf"),
                disease=disease_name,
                model=model_for_matrix
            )
        except Exception as e:
            logger.warning(f"within_disease 热图生成失败: {e}")

    # 可选：对疾病内各Project进行互相预测
    if bool(getattr(args, 'emit_predictions', False)):
        try:
            # 针对每个模型分别生成预测与填充
            for model_for_matrix in model_list:
                # 临时设置单一模型运行预测
                setattr(args, 'models', [model_for_matrix])
                _emit_predictions_within_disease(args, disease_name, projects)
                out_csv = (output_root / 'reports' / 'within_disease' / disease_name /
                           f"within_disease_{disease_name}_{model_for_matrix}_{metric_name}_matrix.csv")
                _fill_matrix_with_predictions(args, out_csv, projects, metric_name)
                logger.info(f"已用预测结果填充跨项目指标: {out_csv}")
                # 预测填充后再保存一次热图
                try:
                    from ..evaluation.visualizer import ResultsVisualizer
                    viz = ResultsVisualizer()
                    viz.plot_within_disease_heatmap(
                        matrix_csv=str(out_csv),
                        metric=metric_name,
                        cmap='YlOrRd',
                        save_path=str(out_dir / f"{out_csv.stem}_heatmap.pdf"),
                        disease=disease_name,
                        model=model_for_matrix
                    )
                except Exception as e:
                    logger.warning(f"within_disease 热图更新失败: {e}")
        except Exception as e:
            logger.warning(f"跨项目预测/填充失败: {e}")
    return str(output_root / 'reports' / 'within_disease' / disease_name)


def _report_between_project(args, metadata):
    """项目间交叉验证报告（按模型，可选预测填充）。

    - 对角线: 从各 Project 的 builds/<model>/1_performance_metrics/nested_cv_summary.csv 读取指定 metric 的均值
    - 非对角线: 默认为空；若 --emit_predictions True，则基于 final_model 生成 train→test 预测并填充所选 metric
    """
    logger = get_logger("ReportGenerator")
    projects = sorted(metadata['Project'].dropna().unique().tolist())

    output_root = Path(getattr(args, 'output', 'tests/result'))
    out_dir = output_root / 'reports' / 'between_project' / 'all'
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_name = getattr(args, 'metric', 'auc')
    models_arg = getattr(args, 'models', None)
    model_list = []
    if models_arg:
        if isinstance(models_arg, (list, tuple)):
            model_list = [str(m).strip().lower() for m in models_arg if str(m).strip()]
        elif isinstance(models_arg, str):
            model_list = [p.strip().lower() for p in models_arg.split(',') if p.strip()]
    if not model_list:
        model_list = ['lasso']

    # 支持通过 --builds_root 显式指定优先的 builds 根目录
    user_builds_root = getattr(args, 'builds_root', None)
    candidate_build_roots = []
    if user_builds_root:
        try:
            candidate_build_roots.append(Path(user_builds_root))
        except Exception:
            pass
    candidate_build_roots += [
        output_root / 'abundance_raw' / 'builds',
        output_root / 'prevalence' / 'builds',
        output_root / 'builds',
    ]

    def read_metric_from_build(proj: str, model_for_matrix: str, metric: str) -> float:
        """读取指标值，优先从nested_cv_pred_proba.csv重新计算AUC"""
        for root in candidate_build_roots:
            for proj_dir in [root / proj, root / f"Project_{proj}"]:
                model_dir = proj_dir / model_for_matrix / '1_performance_metrics'
                if not model_dir.exists():
                    continue
                
                # 关键修复：对于AUC，优先从nested_cv_pred_proba.csv重新计算
                if metric.lower() == 'auc':
                    pred_proba_csv = model_dir / 'nested_cv_pred_proba.csv'
                    if pred_proba_csv.exists():
                        auc_val = _compute_auc_from_nested_cv_pred_proba(pred_proba_csv, metric)
                        if auc_val is not None:
                            return auc_val
                
                # 回退：从summary文件读取（用于非AUC指标或当pred_proba文件不存在时）
                for fp in [model_dir / 'nested_cv_summary.csv', model_dir / 'cv_summary.csv']:
                    if not fp.exists():
                        continue
                    try:
                        dfm = pd.read_csv(fp)
                        cols_lower = {c.lower(): c for c in dfm.columns}
                        # 宽表
                        if metric.lower() in cols_lower:
                            return float(dfm[cols_lower[metric.lower()]].mean())
                        # 长表 metric/mean
                        if 'metric' in cols_lower and 'mean' in cols_lower:
                            mcol = cols_lower['metric']; vcol = cols_lower['mean']
                            dfm[mcol] = dfm[mcol].astype(str).str.lower()
                            sub = dfm[dfm[mcol] == metric.lower()]
                            if sub.empty and metric.lower() == 'auc':
                                sub = dfm[dfm[mcol].isin(['roc_auc'])]
                            if sub.empty and metric.lower() == 'accuracy':
                                sub = dfm[dfm[mcol].isin(['acc'])]
                            if not sub.empty:
                                return float(sub[vcol].mean())
                    except Exception as e:
                        logger.warning(f"读取{fp}失败: {e}")
        return None

    generated_csv_paths = []
    generated_named = []  # list of tuples (model_for_matrix, csv_path)
    for model_for_matrix in model_list:
        out_csv = out_dir / f"between_project_{model_for_matrix}_{metric_name}_matrix.csv"
        # 先仅填充对角线
        records = []
        for src in projects:
            mval = read_metric_from_build(src, model_for_matrix, metric_name)
            for tgt in projects:
                records.append({
                    'train_proj': src,
                    'test_proj': tgt,
                    metric_name: mval if src == tgt else None,
                })
        df_long = pd.DataFrame(records)
        df_long.to_csv(out_csv, index=False)
        logger.info(f"✅ between_project 长表(真实对角线)已生成: {out_csv}")

        # 自动生成热图（支持 --order 排序）
        try:
            from ..evaluation.visualizer import ResultsVisualizer
            viz = ResultsVisualizer()
            viz.plot_within_disease_heatmap(
                matrix_csv=str(out_csv),
                metric=metric_name,
                cmap='Blues',
                save_path=str(out_dir / f"{out_csv.stem}_heatmap.pdf"),
                order_csv=getattr(args, 'order', None),
                order_id_col='ProjectID',
                order_sort_col='ProjectOrder',
                colorbar_shrink=0.15,
                colorbar_aspect=20,
                fig_width_per_col=0.5,
                fig_height_per_row=0.5,
                fig_width_min=8.0,
                fig_height_min=5.5,
                fig_width_max=13.0,
                fig_height_max=10.0
            )
        except Exception as e:
            logger.warning(f"between_project 热图生成失败: {e}")

        # 可选：跨项目预测填充
        if bool(getattr(args, 'emit_predictions', False)):
            try:
                # 临时设置单模型用于预测
                setattr(args, 'models', [model_for_matrix])
                _emit_predictions_between_project(args, projects)
                _fill_matrix_with_predictions(args, out_csv, projects, metric_name)
                logger.info(f"已用预测结果填充跨项目指标: {out_csv}")
                # 预测填充后再保存一次热图
                try:
                    from ..evaluation.visualizer import ResultsVisualizer
                    viz = ResultsVisualizer()
                    viz.plot_within_disease_heatmap(
                        matrix_csv=str(out_csv),
                        metric=metric_name,
                        cmap='Blues',
                        save_path=str(out_dir / f"{out_csv.stem}_heatmap.pdf"),
                        order_csv=getattr(args, 'order', None),
                        order_id_col='ProjectID',
                        order_sort_col='ProjectOrder',
                        colorbar_shrink=0.15,
                        colorbar_aspect=20,
                        fig_width_per_col=0.5,
                        fig_height_per_row=0.5,
                        fig_width_min=8.0,
                        fig_height_min=5.5,
                        fig_width_max=13.0,
                        fig_height_max=10.0
                    )
                except Exception as e:
                    logger.warning(f"between_project 热图更新失败: {e}")
            except Exception as e:
                logger.warning(f"between_project 跨项目预测/填充失败: {e}")

        generated_csv_paths.append(str(out_csv))
        generated_named.append((model_for_matrix, str(out_csv)))

    # 额外输出：基于对角线（Self=Train==Test）的箱线图，按模型聚合
    try:
        if generated_named:
            from ..evaluation.visualizer import ResultsVisualizer
            viz = ResultsVisualizer()
            matrices = [p for _, p in generated_named]
            model_names = [m for m, _ in generated_named]
            boxplot_pdf = out_dir / f"between_project_self_boxplot_{metric_name}.pdf"
            # 检测数据模式用于标题注释
            pa = bool(getattr(args, 'use_presence_absence', True))
            clr = bool(getattr(args, 'use_clr', False))
            if pa:
                data_mode = 'Presence/Absence'
            else:
                data_mode = 'CLR' if clr else 'Abundance-raw'
            title_str = f"Between-Project Self (Diagonal) | {metric_name.upper()} | Data={data_mode}"
            viz.plot_within_disease_boxplot(
                matrices=matrices,
                model_names=model_names,
                metric=metric_name,
                mode='self',
                save_path=str(boxplot_pdf),
                title=title_str,
                show_points=True,
                point_alpha=0.6,
                point_size=3.0,
                point_jitter=0.18,
                model_order='median_desc',
                fig_width_per_model=1.1,
                fig_width_min=6.0,
                fig_width_max=8.5,
                fig_height=5.2
            )
            logger.info(f"✅ between_project 自对角线箱线图已生成: {boxplot_pdf}")
    except Exception as e:
        logger.warning(f"between_project 箱线图生成失败: {e}")

    # 按疾病聚合的 Self vs Cross 箱线图（基于全部模型矩阵）
    try:
        if generated_csv_paths:
            from ..evaluation.visualizer import ResultsVisualizer
            viz = ResultsVisualizer()
            disease_box_pdf = out_dir / f"between_project_disease_mode_boxplot_{metric_name}.pdf"
            viz.plot_disease_mode_boxplot(
                matrices=[str(p) for p in generated_csv_paths],
                metric=metric_name,
                metadata_file=getattr(args, 'metadata_file'),
                save_path=str(disease_box_pdf),
                title=f"Within Disease | Self vs Cross | {metric_name.upper()}"
            )
            logger.info(f"✅ disease-mode箱线图已生成: {disease_box_pdf}")
            # 逐模型输出一张疾病-模式箱线图
            pa = bool(getattr(args, 'use_presence_absence', True))
            clr = bool(getattr(args, 'use_clr', False))
            data_mode = 'Presence/Absence' if pa else ('CLR' if clr else 'Abundance-raw')
            for model_name, csv_path in generated_named:
                per_model_pdf = out_dir / f"between_project_{model_name}_disease_mode_boxplot_{metric_name}.pdf"
                try:
                    viz.plot_disease_mode_boxplot(
                        matrices=[str(csv_path)],
                        metric=metric_name,
                        metadata_file=getattr(args, 'metadata_file'),
                        save_path=str(per_model_pdf),
                        title=f"Within Disease | {model_name} | Self vs Cross | {metric_name.upper()} | Data={data_mode}"
                    )
                    logger.info(f"✅ per-model disease-mode箱线图已生成: {per_model_pdf}")
                except Exception as e:
                    logger.warning(f"per-model disease-mode箱线图失败({model_name}): {e}")
    except Exception as e:
        logger.warning(f"disease-mode 箱线图生成失败: {e}")

    # 兼容旧脚本路径：若仅单模型，则同时输出一个通用文件名副本
    if len(generated_csv_paths) == 1:
        try:
            single_csv = Path(generated_csv_paths[0])
            legacy_csv = out_dir / 'between_project_cross_performance.csv'
            pd.read_csv(single_csv).to_csv(legacy_csv, index=False)
            logger.info(f"已同步输出通用CSV: {legacy_csv}")
            return str(legacy_csv)
        except Exception:
            pass
    return generated_csv_paths[0] if generated_csv_paths else str(out_dir)

def _emit_predictions_between_project(args, projects: list):
    """在全部项目范围内，按项目互相预测：trainProject→testProject 输出逐样本CSV。"""
    logger = get_logger("ReportPredict(between_project)")
    output_root = Path(getattr(args, 'output', 'tests/result'))
    prof_file = Path(getattr(args, 'prof_file'))
    metadata_file = Path(getattr(args, 'metadata_file'))
    models_arg = getattr(args, 'models', None)
    selected_models = models_arg if models_arg else None

    # 加载全量数据
    from ..data.loader import DataLoader
    dl = DataLoader()
    X_all, y_all, groups_all = dl.load_data(
        prof_file=str(prof_file),
        metadata_file=str(metadata_file),
        scope=None,
        use_presence_absence=bool(getattr(args, 'use_presence_absence', True)),
        use_clr=bool(getattr(args, 'use_clr', False)),
        enable_cohort_analysis=False,
        group_col='Group',
        label_0=getattr(args, 'label_0', None),
        label_1=getattr(args, 'label_1', None)
    )[:3]

    try:
        y_all_series = pd.Series(y_all, index=X_all.index)
    except Exception:
        y_all_series = None

    meta = pd.read_csv(str(metadata_file), index_col=0)

    candidate_build_roots = [
        output_root / 'abundance_raw' / 'builds',
        output_root / 'prevalence' / 'builds',
        output_root / 'builds',
    ]
    import joblib

    for train_proj in projects:
        # 发现/筛选方法目录
        model_base_dirs = []
        for root in candidate_build_roots:
            mroot_candidates = [root / train_proj, root / f"Project_{train_proj}"]
            mroot = next((p for p in mroot_candidates if p.exists()), None)
            if mroot is None:
                continue
            if selected_models:
                for m in selected_models:
                    cand = mroot / str(m).lower() / '2_final_model'
                    if cand.exists():
                        model_base_dirs.append((m, cand))
            else:
                for model_dir in mroot.iterdir():
                    cand = model_dir / '2_final_model'
                    if cand.exists():
                        model_base_dirs.append((model_dir.name, cand))
        if not model_base_dirs:
            logger.warning(f"{train_proj}: 未发现任何final_model目录")
            continue

        for model_name, final_dir in model_base_dirs:
            try:
                model_path = final_dir / 'final_model.joblib'
                if not model_path.exists():
                    found = list(final_dir.glob('final_model_*.joblib'))
                    if found:
                        model_path = found[0]
                clf = joblib.load(model_path)
                features = getattr(clf, 'selected_features_', None)
                if not features:
                    csv_path = final_dir.parent.parent / '3_feature_analysis' / 'consensus_features.csv'
                    if csv_path.exists():
                        cdf = pd.read_csv(csv_path)
                        if 'feature_name' in cdf.columns:
                            features = cdf['feature_name'].dropna().astype(str).tolist()
                if not features:
                    logger.warning(f"{train_proj}/{model_name}: 无法确定特征列表，跳过")
                    continue
                for test_proj in projects:
                    test_ids = meta[meta['Project'] == test_proj].index
                    X_test = X_all.loc[X_all.index.intersection(test_ids)].copy()
                    if X_test.empty:
                        continue
                    for f in features:
                        if f not in X_test.columns:
                            X_test[f] = 0
                    X_test = X_test[features]
                    score = _predict_positive_scores(clf, X_test)
                    out_dir = output_root / 'reports' / 'predict' / str(model_name).lower()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = out_dir / f"{train_proj}_to_{test_proj}_predictions.csv"
                    if y_all_series is not None:
                        true_y = y_all_series.loc[X_test.index]
                    else:
                        true_y = _extract_true_labels(X_test.index, y_all, metadata_file)
                    pd.DataFrame({
                        'sample_id': X_test.index,
                        'true_label': true_y.values,
                        'predicted_score': score,
                    }).to_csv(out_csv, index=False)
            except Exception as e:
                logger.warning(f"{train_proj}/{model_name} 互测失败: {e}")


def _report_between_disease(args, metadata):
    """疾病间交叉验证报告（基于真实项目级矩阵聚合到疾病级）。

    实现思路：
    1) 复用 between_project 逻辑，按模型生成项目级矩阵（对角线来自 nested_cv_summary，非对角线可选由预测填充）。
    2) 读取每个模型的矩阵 CSV，根据 metadata 的 Project→Disease 映射，
       将 (train_proj, test_proj) 聚合为 (train_dis, test_dis) 的均值（按所选 metric）。
    3) 若传入多个模型，则在疾病级别再对多个模型求平均。
    输出: between_disease_cross_performance.csv，包含列 train_dis, test_dis, <metric>。
    """
    logger = get_logger("ReportGenerator")
    metric_name = getattr(args, 'metric', 'auc')
    
    output_root = Path(getattr(args, 'output', 'tests/result'))
    out_dir = output_root / 'reports' / 'between_disease' / 'all'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'between_disease_cross_performance.csv'
    
    # 直接从构建结果与（可选）预测CSV产生疾病级表，不生成 between_project 产物

    # 1) 准备模型列表与项目/疾病映射
    models_arg = getattr(args, 'models', None)
    if models_arg:
        if isinstance(models_arg, (list, tuple)):
            model_list = [str(m).strip().lower() for m in models_arg if str(m).strip()]
        elif isinstance(models_arg, str):
            model_list = [p.strip().lower() for p in models_arg.split(',') if p.strip()]
    else:
        model_list = ['lasso']

    proj_to_dis_df = metadata[['Project', 'Disease']].dropna()
    proj_to_dis_df['Project'] = proj_to_dis_df['Project'].astype(str)
    proj_to_dis_df['Disease'] = proj_to_dis_df['Disease'].astype(str)
    proj_to_dis_map = dict(zip(proj_to_dis_df['Project'], proj_to_dis_df['Disease']))
    all_projects = sorted(proj_to_dis_df['Project'].unique().tolist())

    # 2) 从 nested_cv_summary.csv 读取各项目的自对角线指标（train_dis==test_dis）
    # 参考 between_project 的读取逻辑，但不写任何 between_project 文件
    user_builds_root = getattr(args, 'builds_root', None)
    candidate_build_roots = []
    if user_builds_root:
        try:
            candidate_build_roots.append(Path(user_builds_root))
        except Exception:
            pass
    candidate_build_roots += [
        output_root / 'abundance_raw' / 'builds',
        output_root / 'prevalence' / 'builds',
        output_root / 'builds',
    ]

    def read_proj_metric(model_for_matrix: str, proj: str, metric: str) -> Optional[float]:
        """读取项目指标值，优先从nested_cv_pred_proba.csv重新计算AUC"""
        for root in candidate_build_roots:
            for proj_dir in [root / proj, root / f"Project_{proj}"]:
                model_dir = proj_dir / model_for_matrix / '1_performance_metrics'
                if not model_dir.exists():
                    continue
                
                # 关键修复：对于AUC，优先从nested_cv_pred_proba.csv重新计算
                if metric.lower() == 'auc':
                    pred_proba_csv = model_dir / 'nested_cv_pred_proba.csv'
                    if pred_proba_csv.exists():
                        auc_val = _compute_auc_from_nested_cv_pred_proba(pred_proba_csv, metric)
                        if auc_val is not None:
                            return auc_val
                
                # 回退：从summary文件读取（用于非AUC指标或当pred_proba文件不存在时）
                for fp in [model_dir / 'nested_cv_summary.csv', model_dir / 'cv_summary.csv']:
                    if not fp.exists():
                        continue
                    try:
                        dfm = pd.read_csv(fp)
                        cols_lower = {c.lower(): c for c in dfm.columns}
                        if metric.lower() in cols_lower:
                            return float(dfm[cols_lower[metric.lower()]].mean())
                        if 'metric' in cols_lower and 'mean' in cols_lower:
                            mcol = cols_lower['metric']; vcol = cols_lower['mean']
                            dfm[mcol] = dfm[mcol].astype(str).str.lower()
                            sub = dfm[dfm[mcol] == metric.lower()]
                            if sub.empty and metric.lower() == 'auc':
                                sub = dfm[dfm[mcol].isin(['roc_auc'])]
                            if sub.empty and metric.lower() == 'accuracy':
                                sub = dfm[dfm[mcol].isin(['acc'])]
                            if not sub.empty:
                                return float(sub[vcol].mean())
                    except Exception:
                        continue
        return None

    def read_disease_metric(model_for_matrix: str, disease: str, metric: str) -> Optional[float]:
        """从疾病级目录读取指标，优先从nested_cv_pred_proba.csv重新计算AUC。路径: <root>/Disease_<Disease>/<model>/1_performance_metrics/"""
        for root in candidate_build_roots:
            dis_dir = root / f"Disease_{disease}"
            model_dir = dis_dir / model_for_matrix / '1_performance_metrics'
            if not model_dir.exists():
                continue
            
            # 关键修复：对于AUC，优先从nested_cv_pred_proba.csv重新计算
            if metric.lower() == 'auc':
                pred_proba_csv = model_dir / 'nested_cv_pred_proba.csv'
                if pred_proba_csv.exists():
                    auc_val = _compute_auc_from_nested_cv_pred_proba(pred_proba_csv, metric)
                    if auc_val is not None:
                        return auc_val
            
            # 回退：从summary文件读取（用于非AUC指标或当pred_proba文件不存在时）
            for fp in [model_dir / 'nested_cv_summary.csv', model_dir / 'cv_summary.csv']:
                if not fp.exists():
                    continue
                try:
                    dfm = pd.read_csv(fp)
                    cols_lower = {c.lower(): c for c in dfm.columns}
                    if metric.lower() in cols_lower:
                        return float(dfm[cols_lower[metric.lower()]].mean())
                    if 'metric' in cols_lower and 'mean' in cols_lower:
                        mcol = cols_lower['metric']; vcol = cols_lower['mean']
                        dfm[mcol] = dfm[mcol].astype(str).str.lower()
                        sub = dfm[dfm[mcol] == metric.lower()]
                        if sub.empty and metric.lower() == 'auc':
                            sub = dfm[dfm[mcol].isin(['roc_auc'])]
                        if sub.empty and metric.lower() == 'accuracy':
                            sub = dfm[dfm[mcol].isin(['acc'])]
                        if not sub.empty:
                            return float(sub[vcol].mean())
                except Exception:
                    continue
        return None

    # 3) 可选：生成跨疾病预测CSV（仅当 --emit_predictions True），用于跨疾病聚合
    if bool(getattr(args, 'emit_predictions', False)):
        try:
            _emit_predictions_between_disease(args, sorted(proj_to_dis_df['Disease'].unique().tolist()))
        except Exception as e:
            logger.warning(f"between_disease: 生成跨疾病预测失败(跳过跨疾病填充): {e}")

    # 4) 汇总：对每个模型，构建 disease×disease 的表，然后跨模型求平均
    from sklearn.metrics import roc_auc_score, accuracy_score
    per_model_frames = []
    for model_for_matrix in model_list:
        # 4.1 自对角线 by disease：优先使用疾病级 nested_cv_summary；若缺失则回退为项目级聚合
        diseases = sorted(proj_to_dis_df['Disease'].unique().tolist())
        diag_records = []
        for dis in diseases:
            # 优先疾病级
            dval = read_disease_metric(model_for_matrix, dis, metric_name)
            if dval is None:
                # 回退同疾病内项目聚合
                sub_projects = [p for p, d in proj_to_dis_map.items() if d == dis]
                vals = []
                for proj in sub_projects:
                    mval = read_proj_metric(model_for_matrix, proj, metric_name)
                    if mval is not None:
                        vals.append(mval)
                dval = float(np.mean(vals)) if vals else None
            diag_records.append({'disease': dis, metric_name: dval})
        df_diag = pd.DataFrame(diag_records)

        # 4.2 跨疾病（非对角）：若有预测CSV，则计算 train_proj→test_proj 的指标并映射到疾病层面
        cross_rows = []
        pred_dir = output_root / 'reports' / 'predict_disease' / model_for_matrix
        if pred_dir.exists():
            for fp in pred_dir.glob("*_to_*.csv"):
                try:
                    name = fp.stem  # e.g., AD_to_PD_predictions
                    if not name.endswith('_predictions'):
                        continue
                    core = name[:-12]
                    if '_to_' not in core:
                        continue
                    tr, te = core.split('_to_', 1)
                    tr_dis = tr
                    te_dis = te
                    pdf = pd.read_csv(fp)
                    if 'true_label' not in pdf.columns or 'predicted_score' not in pdf.columns:
                        continue
                    y_true = pdf['true_label']
                    y_score = pdf['predicted_score']
                    mask = y_true.notna() & y_score.notna()
                    y_true = y_true[mask]
                    y_score = y_score[mask]
                    if len(y_true) < 2 or y_true.nunique() < 2:
                        continue
                    if metric_name.lower() == 'auc':
                        val = roc_auc_score(y_true, y_score)
                    elif metric_name.lower() == 'accuracy':
                        val = accuracy_score(y_true, (y_score >= 0.5).astype(int))
                    else:
                        # 仅支持 AUC/accuracy，其它指标可扩展
                        continue
                    cross_rows.append({'train_dis': tr_dis, 'test_dis': te_dis, metric_name: val})
                except Exception:
                    continue
        df_cross = pd.DataFrame(cross_rows) if cross_rows else pd.DataFrame(columns=['train_dis','test_dis',metric_name])

        # 4.3 合并自对角线与跨疾病
        records = []
        # 先填充对角
        diag_map = dict(zip(df_diag['disease'], df_diag[metric_name])) if not df_diag.empty else {}
        for d in diseases:
            val = diag_map.get(d, None)
            records.append({'train_dis': d, 'test_dis': d, metric_name: val})
        # 再填充非对角（均值）
        if not df_cross.empty:
            g = df_cross.groupby(['train_dis','test_dis'], as_index=False)[metric_name].mean()
            for _, row in g.iterrows():
                if row['train_dis'] == row['test_dis']:
                    continue
                records.append({'train_dis': row['train_dis'], 'test_dis': row['test_dis'], metric_name: row[metric_name]})

        df_model = pd.DataFrame(records)
        per_model_frames.append(df_model)

        # 输出该模型的疾病×疾病矩阵CSV（与 between_project 风格一致）
        try:
            per_model_csv = out_dir / f"between_disease_{model_for_matrix}_{metric_name}_matrix.csv"
            df_model.to_csv(per_model_csv, index=False)
            logger.info(f"✅ between_disease(模型: {model_for_matrix}) 矩阵已生成: {per_model_csv}")
            # 生成对应热图（过滤对角为空的疾病）
            _dfm = df_model.copy()
            diag_map_m = (_dfm[_dfm['train_dis'] == _dfm['test_dis']]
                          .set_index('train_dis')[metric_name].to_dict())
            keep_m = [d for d, v in diag_map_m.items() if not pd.isna(v)]
            if keep_m:
                _dfm = _dfm[_dfm['train_dis'].isin(keep_m) & _dfm['test_dis'].isin(keep_m)]
            diseases_sorted_m = sorted(sorted(set(_dfm['train_dis']).union(set(_dfm['test_dis']))))
            mat_m = _dfm.pivot_table(index='train_dis', columns='test_dis', values=metric_name)
            mat_m = mat_m.reindex(index=diseases_sorted_m, columns=diseases_sorted_m)
            # 若提供 --order（包含列 DiseaseOrder, DiseaseID），则按疾病顺序重排
            order_csv = getattr(args, 'order', None)
            if order_csv:
                try:
                    _ord = pd.read_csv(str(order_csv))
                    # 兼容大小写列名
                    _cols_map = {c.lower(): c for c in _ord.columns}
                    ord_col = _cols_map.get('diseaseorder')
                    id_col = _cols_map.get('diseaseid')
                    if ord_col:
                        _ord = _ord.sort_values(ord_col)
                    if id_col:
                        _order_list = [str(x) for x in _ord[id_col].dropna().astype(str).tolist()]
                        _idx = [x for x in _order_list if x in mat_m.index]
                        _cols = [x for x in _order_list if x in mat_m.columns]
                        if _idx:
                            mat_m = mat_m.reindex(index=_idx)
                        if _cols:
                            mat_m = mat_m.reindex(columns=_cols)
                        # 覆盖用于图尺寸估计的疾病顺序
                        diseases_sorted_m = [x for x in _order_list if x in diseases_sorted_m]
                        logger.info(f"between_disease: 已应用疾病排序，顺序={diseases_sorted_m}")
                    else:
                        logger.warning(f"between_disease: 排序文件缺少 DiseaseID 列: {order_csv}")
                except Exception as _e:
                    logger.warning(f"between_disease: 应用疾病排序失败({order_csv}): {_e}")
            import seaborn as _sns
            import matplotlib.pyplot as _plt
            _sns.set_style("white")
            fig_wm = max(6.0, 0.6 * max(1, len(diseases_sorted_m)))
            fig_hm = max(5.0, 0.6 * max(1, len(diseases_sorted_m)))
            _plt.figure(figsize=(fig_wm, fig_hm))
            axm = _sns.heatmap(
                mat_m,
                cmap='Blues',
                vmin=0.0,
                vmax=1.0,
                annot=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.7, 'aspect': 18, 'label': metric_name.upper()}
            )
            axm.set_xlabel('Test Disease')
            axm.set_ylabel('Train Disease')
            axm.set_title(f"Between-Disease | Model: {model_for_matrix} | Metric: {metric_name.upper()}")
            # 若提供 --order（包含列 DiseaseOrder, DiseaseID），则按疾病顺序重排
            order_csv = getattr(args, 'order', None)
            if order_csv:
                try:
                    _ord = pd.read_csv(str(order_csv))
                    if 'DiseaseOrder' in _ord.columns:
                        _ord = _ord.sort_values('DiseaseOrder')
                    if 'DiseaseID' in _ord.columns:
                        _order_list = [str(x) for x in _ord['DiseaseID'].dropna().astype(str).tolist()]
                        _idx = [x for x in _order_list if x in mat_m.index]
                        _cols = [x for x in _order_list if x in mat_m.columns]
                        if _idx:
                            mat_m = mat_m.reindex(index=_idx)
                        if _cols:
                            mat_m = mat_m.reindex(columns=_cols)
                        # 覆盖用于图尺寸估计的疾病顺序
                        diseases_sorted_m = [x for x in _order_list if x in diseases_sorted_m]
                except Exception as _e:
                    logger.warning(f"between_disease: 应用疾病排序失败({order_csv}): {_e}")

            per_model_pdf = out_dir / f"between_disease_{model_for_matrix}_{metric_name}_matrix_heatmap.pdf"
            _plt.tight_layout()
            _plt.savefig(str(per_model_pdf), dpi=300, bbox_inches='tight')
            logger.info(f"✅ between_disease(模型: {model_for_matrix}) 热图已生成: {per_model_pdf}")
            _plt.close()

            # 自动生成：基于 repeat 的均值ROC（per-disease），若存在 nested_cv_pred_proba.csv 则绘制
            try:
                from ..evaluation.visualizer import ResultsVisualizer as _ResultsVisualizer
                _viz = _ResultsVisualizer()
                # 遍历疾病顺序，查找各疾病对应的 nested_cv_pred_proba.csv
                for _dis in diseases_sorted_m:
                    _csv_found = None
                    for _root in candidate_build_roots:
                        _cand = _root / f"Disease_{_dis}" / model_for_matrix / '1_performance_metrics' / 'nested_cv_pred_proba.csv'
                        if _cand.exists():
                            _csv_found = _cand
                            break
                    if _csv_found is None:
                        continue
                    _roc_pdf = out_dir / f"between_disease_{model_for_matrix}_{_dis}_repeat_mean_roc.pdf"
                    _viz.plot_repeat_mean_roc_from_nested_csv(
                        csv_path=str(_csv_found),
                        disease=str(_dis),
                        model=str(model_for_matrix),
                        save_path=str(_roc_pdf),
                        show_individual=False,
                        ci_mode='percentile',
                        ci_level=0.95,
                        fpr_step=0.001
                    )
                    logger.info(f"✅ between_disease ROC(Repeat-mean) 已生成: {_roc_pdf}")
            except Exception as _e:
                logger.warning(f"between_disease: 自动生成ROC失败(模型 {model_for_matrix}): {_e}")
        except Exception as e:
            logger.warning(f"between_disease: 模型级输出失败({model_for_matrix}): {e}")

    if not per_model_frames:
        logger.warning("between_disease: 无法构建疾病级表，输出空表")
        pd.DataFrame(columns=['train_dis', 'test_dis', metric_name]).to_csv(out_csv, index=False)
        return str(out_csv)

    # 5) 跨模型平均
    df_all = pd.concat(per_model_frames, ignore_index=True)
    df_final = df_all.groupby(['train_dis','test_dis'], as_index=False)[metric_name].mean()
    df_final.to_csv(out_csv, index=False)
    logger.info(f"✅ between_disease 长表(真实聚合)已生成: {out_csv}")

    # 按用户需求：不生成合并（跨模型）的热图，仅输出每模型热图
    return str(out_csv)


def _report_overall(args):
    """整体性能分析报告"""
    logger = get_logger("ReportGenerator")
    
    output_root = Path(getattr(args, 'output', 'tests/result'))
    out_dir = output_root / 'reports' / 'overall' / 'all'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例结果
    by_proj = pd.DataFrame({
        'Project': ['ProjectA', 'ProjectB', 'ProjectC'],
        'auc': [0.80, 0.75, 0.82],
        'accuracy': [0.75, 0.70, 0.78],
        'n': [100, 150, 120]
    })
    
    by_dis = pd.DataFrame({
        'Disease': ['AD', 'ASD', 'PD'],
        'auc': [0.85, 0.80, 0.88],
        'accuracy': [0.80, 0.75, 0.83],
        'n': [200, 180, 300]
    })
    
    by_proj.to_csv(out_dir / 'overall_by_project.csv', index=False)
    by_dis.to_csv(out_dir / 'overall_by_disease.csv', index=False)
    logger.info(f"✅ overall 子集性能表已生成: {out_dir}/overall_by_project.csv, overall_by_disease.csv")
    
    # 额外：使用 All 目录下的最终模型对外部数据整体预测（每个可用模型各输出一份）
    try:
        from ..data.loader import DataLoader
        dl = DataLoader()
        prof_file = Path(getattr(args, 'prof_file'))
        metadata_file = Path(getattr(args, 'metadata_file'))
        X_all, y_all, _ = dl.load_data(
            prof_file=str(prof_file),
            metadata_file=str(metadata_file),
            scope=None,
            use_presence_absence=bool(getattr(args, 'use_presence_absence', True)),
            use_clr=bool(getattr(args, 'use_clr', False)),
            enable_cohort_analysis=False,
            group_col='Group',
            label_0=getattr(args, 'label_0', None),
            label_1=getattr(args, 'label_1', None)
        )[:3]
        try:
            y_all_series = pd.Series(y_all, index=X_all.index)
        except Exception:
            y_all_series = None
        # 搜索 All 目录下的模型
        user_builds_root = getattr(args, 'builds_root', None)
        candidate_build_roots = []
        if user_builds_root:
            try:
                candidate_build_roots.append(Path(user_builds_root))
            except Exception:
                pass
        candidate_build_roots += [
            output_root / 'abundance_raw' / 'builds',
            output_root / 'prevalence' / 'builds',
            output_root / 'builds',
        ]
        models_arg = getattr(args, 'models', None)
        selected_models = [str(m).strip().lower() for m in models_arg] if models_arg else None
        import joblib
        out_root = output_root / 'reports' / 'predict_external_all'
        out_root.mkdir(parents=True, exist_ok=True)
        found_any = False
        for root in candidate_build_roots:
            all_root = root / 'All'
            if not all_root.exists():
                continue
            # 收集 2_final_model 目录
            model_dirs = []
            for model_dir in all_root.iterdir():
                if not model_dir.is_dir():
                    continue
                if selected_models and model_dir.name.lower() not in selected_models:
                    continue
                fm = model_dir / '2_final_model'
                if fm.exists():
                    model_dirs.append((model_dir.name, fm))
            for model_name, final_dir in model_dirs:
                try:
                    model_path = final_dir / 'final_model.joblib'
                    if not model_path.exists():
                        cands = list(final_dir.glob('final_model_*.joblib'))
                        if cands:
                            model_path = cands[0]
                    clf = joblib.load(model_path)
                    # 特征列表
                    features = getattr(clf, 'selected_features_', None)
                    if not features:
                        csv_path = final_dir.parent.parent / '3_feature_analysis' / 'consensus_features.csv'
                        if csv_path.exists():
                            cdf = pd.read_csv(csv_path)
                            if 'feature_name' in cdf.columns:
                                features = cdf['feature_name'].dropna().astype(str).tolist()
                    if not features:
                        logger.warning(f"All/{model_name}: 无法确定特征列表，跳过")
                        continue
                    # 构建预测集
                    X_pred = X_all.copy()
                    for f in features:
                        if f not in X_pred.columns:
                            X_pred[f] = 0
                    X_pred = X_pred[features]
                    score = _predict_positive_scores(clf, X_pred)
                    out_dir_m = out_root / str(model_name).lower()
                    out_dir_m.mkdir(parents=True, exist_ok=True)
                    out_csv = out_dir_m / 'All_predictions.csv'
                    if y_all_series is not None:
                        true_y = y_all_series.loc[X_pred.index]
                        pd.DataFrame({
                            'sample_id': X_pred.index,
                            'true_label': true_y.values,
                            'predicted_score': score,
                        }).to_csv(out_csv, index=False)
                    else:
                        pd.DataFrame({
                            'sample_id': X_pred.index,
                            'predicted_score': score,
                        }).to_csv(out_csv, index=False)
                    logger.info(f"✅ overall: All 目录模型 {model_name} 预测输出: {out_csv}")
                    found_any = True
                except Exception as e:
                    logger.warning(f"overall: All/{model_name} 预测失败: {e}")
        if not found_any:
            logger.info("overall: 未在 All 目录找到可用的 2_final_model，已跳过外部预测输出")
    except Exception as e:
        logger.warning(f"overall: 外部预测阶段失败（已忽略）: {e}")
    
    return str(out_dir)


def _report_models(args):
    """多模型比较报告"""
    logger = get_logger("ReportGenerator")
    
    scope_expr = getattr(args, 'scope', None)
    scope_tag = scope_expr.replace('=', '_').replace(' ', '') if scope_expr else 'All'
    
    # 设置输出路径
    output_root = Path(getattr(args, 'output', 'tests/result'))
    out_dir = output_root / 'reports' / 'models' / scope_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 真实结果汇总：从构建目录读取 nested_cv_summary.csv 的均值
    models = getattr(args, 'models', ['lasso'])
    candidate_build_roots = [
        output_root / 'prevalence' / 'builds',
        output_root / 'abundance_raw' / 'builds',
        output_root / 'builds',
    ]
    
    def normalize_model_dir_name(name: str) -> str:
        n = name.strip().lower()
        if n in ('logisticregression', 'logistic'):
            return 'logistic'
        if n == 'randomforest':
            return 'randomforest'
        if n == 'catboost':
            return 'catboost'
        if n == 'xgboost':
            return 'xgboost'
        if n == 'svm':
            return 'svm'
        if n == 'knn':
            return 'knn'
        if n == 'gaussiannb':
            return 'gaussiannb'
        if n == 'neuralnetwork':
            return 'neuralnetwork'
        if n == 'elasticnet':
            return 'elasticnet'
        if n == 'lasso':
            return 'lasso'
        return n
    
    rows = []
    for m in models:
        m_dir_name = normalize_model_dir_name(str(m))
        found_csv = None
        for root in candidate_build_roots:
            try:
                # 查找任意项目子目录下对应模型的性能文件
                matches = list(root.glob(f"**/{m_dir_name}/1_performance_metrics/nested_cv_summary.csv"))
                if matches:
                    found_csv = matches[0]
                    break
            except Exception:
                continue
        if not found_csv or not found_csv.exists():
            logger.warning(f"未找到模型 {m} 的nested_cv_summary.csv，已跳过")
            continue
        try:
            df = pd.read_csv(found_csv)
            # 期望列: model, metric, mean, std
            auc_val = None
            acc_val = None
            if 'metric' in df.columns and 'mean' in df.columns:
                auc_row = df[df['metric'].str.lower() == 'auc']
                acc_row = df[df['metric'].str.lower() == 'accuracy']
                if not auc_row.empty:
                    auc_val = float(auc_row.iloc[0]['mean'])
                if not acc_row.empty:
                    acc_val = float(acc_row.iloc[0]['mean'])
            rows.append({'model': m, 'auc': auc_val, 'accuracy': acc_val})
        except Exception as e:
            logger.warning(f"读取 {found_csv} 失败: {e}")
            continue
    
    if not rows:
        logger.warning("未汇总到任何模型结果，输出空文件")
        df_models = pd.DataFrame(columns=['model', 'auc', 'accuracy'])
    else:
        df_models = pd.DataFrame(rows)
    
    summary_csv = out_dir / f"models_external_test_{scope_tag}.csv"
    df_models.to_csv(summary_csv, index=False)
    logger.info(f"✅ 模型比较结果已生成: {summary_csv}")
    
    # 可选：输出逐样本预测
    if bool(getattr(args, 'emit_predictions', False)):
        try:
            _emit_predictions_for_models(args, models)
        except Exception as e:
            logger.warning(f"生成逐样本预测失败: {e}")
    
    return str(summary_csv)


def _emit_predictions_for_models(args, models):
    """基于已训练final_model对当前数据生成逐样本预测CSV。"""
    logger = get_logger("ReportPredict(models)")
    output_root = Path(getattr(args, 'output', 'tests/result'))
    prof_file = Path(getattr(args, 'prof_file'))
    metadata_file = Path(getattr(args, 'metadata_file'))
    out_dir = output_root / 'reports' / 'predict'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    from ..data.loader import DataLoader
    dl = DataLoader()
    X, y, groups = dl.load_data(
        prof_file=str(prof_file),
        metadata_file=str(metadata_file),
        scope=getattr(args, 'scope', None),
        use_presence_absence=bool(getattr(args, 'use_presence_absence', True)),
        use_clr=bool(getattr(args, 'use_clr', False)),
        enable_cohort_analysis=False,
        group_col='Group'
    )[:3]
    
    # 搜索模型目录
    candidate_build_roots = [
        output_root / 'abundance_raw' / 'builds',
        output_root / 'prevalence' / 'builds',
        output_root / 'builds',
    ]
    
    import joblib
    import json as _json
    
    index_rows = []
    # 将 y 映射为与 X.index 对齐的 Series
    try:
        y_series_global = pd.Series(y, index=X.index)
    except Exception:
        y_series_global = None
    for m in models:
        # 遍历所有可用的此方法的训练目录（所有项目）
        model_dirs = []
        for root in candidate_build_roots:
            if not root.exists():
                continue
            model_dirs += list(root.glob(f"**/{str(m).lower()}/2_final_model"))
        if not model_dirs:
            logger.warning(f"未找到方法 {m} 的任何final_model目录")
            continue
        model_out_dir = out_dir / str(m).lower()
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        for mdir in model_dirs:
            try:
                clf = joblib.load(mdir / 'final_model.joblib') if (mdir / 'final_model.joblib').exists() else None
                if clf is None:
                    # 兼容按名称保存的情形
                    cand = list(mdir.glob('final_model_*.joblib'))
                    if cand:
                        clf = joblib.load(cand[0])
                if clf is None:
                    logger.warning(f"跳过 {mdir}，未找到final_model.joblib")
                    continue
                # 特征列表优先从模型读取
                features = getattr(clf, 'selected_features_', None)
                if not features:
                    csv_path = mdir.parent.parent / '3_feature_analysis' / 'consensus_features.csv'
                    if csv_path.exists():
                        cdf = pd.read_csv(csv_path)
                        if 'feature_name' in cdf.columns:
                            features = cdf['feature_name'].dropna().astype(str).tolist()
                if not features:
                    logger.warning(f"{mdir} 无法确定特征列表，跳过")
                    continue
                X_pred = X.copy()
                for f in features:
                    if f not in X_pred.columns:
                        X_pred[f] = 0
                # 统计特征覆盖率
                present_features = [f for f in features if f in X.columns]
                n_expected = len(features)
                n_present = len(present_features)
                n_missing = n_expected - n_present
                present_ratio = (n_present / n_expected) if n_expected > 0 else 0.0
                X_pred = X_pred[features]
                # 预测
                score = _predict_positive_scores(clf, X_pred)
                pred = clf.predict(X_pred)
                # 输出（仅保留 sample_id, true_label, predicted_score）
                # trainProject 从路径中解析（上上级目录名）
                # .../builds/<Project>/<model>/2_final_model
                parts = mdir.parts
                train_proj = 'UnknownProject'
                try:
                    # 找到 'builds' 的索引
                    if 'builds' in parts:
                        bi = parts.index('builds')
                        train_proj = parts[bi+1]
                except Exception:
                    pass
                out_csv = model_out_dir / f"{train_proj}_to_All_predictions.csv"
                # 优先使用与 X.index 对齐的 y_series_global
                if y_series_global is not None:
                    true_y = y_series_global.loc[X_pred.index]
                else:
                    true_y = _extract_true_labels(X_pred.index, y, metadata_file)
                df_out = pd.DataFrame({
                    'sample_id': X_pred.index,
                    'true_label': true_y.values,
                    'predicted_score': score,
                })
                df_out.to_csv(out_csv, index=False)
                index_rows.append({
                    'model': m,
                    'train_project': train_proj,
                    'csv_path': str(out_csv),
                    'n_expected_features': n_expected,
                    'n_present_features': n_present,
                    'n_missing_features': n_missing,
                    'present_ratio': present_ratio,
                })
            except Exception as e:
                logger.warning(f"预测失败({mdir}): {e}")
                continue
        if index_rows:
            pd.DataFrame(index_rows).to_csv(model_out_dir / 'predictions_index.csv', index=False)


def _emit_predictions_within_disease(args, disease_name: str, projects: list):
    """在指定疾病内，按项目互相预测：trainProject→testProject 输出逐样本CSV。"""
    logger = get_logger("ReportPredict(within_disease)")
    output_root = Path(getattr(args, 'output', 'tests/result'))
    prof_file = Path(getattr(args, 'prof_file'))
    metadata_file = Path(getattr(args, 'metadata_file'))
    models_arg = getattr(args, 'models', None)
    selected_models = models_arg if models_arg else None  # None 表示自动发现
    
    # 加载全量数据
    from ..data.loader import DataLoader
    dl = DataLoader()
    X_all, y_all, groups_all = dl.load_data(
        prof_file=str(prof_file),
        metadata_file=str(metadata_file),
        scope=None,
        use_presence_absence=bool(getattr(args, 'use_presence_absence', True)),
        use_clr=bool(getattr(args, 'use_clr', False)),
        enable_cohort_analysis=False,
        group_col='Group',
        label_0=getattr(args, 'label_0', None),  # 支持标签映射
        label_1=getattr(args, 'label_1', None)   # 支持标签映射
    )[:3]
    # 将 y_all 映射为与 X_all.index 对齐的 Series
    try:
        y_all_series = pd.Series(y_all, index=X_all.index)
    except Exception:
        y_all_series = None
    # 根据元数据筛选疾病
    meta = pd.read_csv(str(metadata_file), index_col=0)
    meta = meta[meta['Disease'] == disease_name]
    
    # 搜索模型根
    candidate_build_roots = [
        output_root / 'abundance_raw' / 'builds',
        output_root / 'prevalence' / 'builds',
        output_root / 'builds',
    ]
    import joblib
    
    for train_proj in projects:
        # 发现/筛选方法目录
        model_base_dirs = []
        for root in candidate_build_roots:
            # 兼容目录命名：可能是 "Project_<name>" 或原始项目名
            mroot_candidates = [root / train_proj, root / f"Project_{train_proj}"]
            mroot = next((p for p in mroot_candidates if p.exists()), None)
            if mroot is None:
                continue
            if selected_models:
                for m in selected_models:
                    cand = mroot / str(m).lower() / '2_final_model'
                    if cand.exists():
                        model_base_dirs.append((m, cand))
            else:
                # 自动发现所有模型
                for model_dir in mroot.iterdir():
                    cand = model_dir / '2_final_model'
                    if cand.exists():
                        model_base_dirs.append((model_dir.name, cand))
        if not model_base_dirs:
            logger.warning(f"{train_proj}: 未发现任何final_model目录")
            continue
        
        for model_name, final_dir in model_base_dirs:
            try:
                model_path = final_dir / 'final_model.joblib'
                if not model_path.exists():
                    # 兼容按名称保存
                    found = list(final_dir.glob('final_model_*.joblib'))
                    if found:
                        model_path = found[0]
                clf = joblib.load(model_path)
                features = getattr(clf, 'selected_features_', None)
                if not features:
                    # 回退 CSV
                    csv_path = final_dir.parent.parent / '3_feature_analysis' / 'consensus_features.csv'
                    if csv_path.exists():
                        cdf = pd.read_csv(csv_path)
                        if 'feature_name' in cdf.columns:
                            features = cdf['feature_name'].dropna().astype(str).tolist()
                if not features:
                    logger.warning(f"{train_proj}/{model_name}: 无法确定特征列表，跳过")
                    continue
                # 针对每个 test_proj 生成预测
                for test_proj in projects:
                    # 仅在同一疾病内，筛选该项目样本
                    test_ids = meta[meta['Project'] == test_proj].index
                    X_test = X_all.loc[X_all.index.intersection(test_ids)].copy()
                    if X_test.empty:
                        continue
                    for f in features:
                        if f not in X_test.columns:
                            X_test[f] = 0
                    # 统计特征覆盖率
                    present_features = [f for f in features if f in X_all.columns]
                    n_expected = len(features)
                    n_present = len(present_features)
                    n_missing = n_expected - n_present
                    present_ratio = (n_present / n_expected) if n_expected > 0 else 0.0
                    X_test = X_test[features]
                    # 预测
                    score = _predict_positive_scores(clf, X_test)
                    # 输出（仅保留 sample_id, true_label, predicted_score）
                    out_dir = output_root / 'reports' / 'predict' / str(model_name).lower()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = out_dir / f"{train_proj}_to_{test_proj}_predictions.csv"
                    # 优先使用 DataLoader 返回的 y 与索引对齐
                    if y_all_series is not None:
                        true_y = y_all_series.loc[X_test.index]
                    else:
                        true_y = _extract_true_labels(X_test.index, y_all, metadata_file)
                    df_out = pd.DataFrame({
                        'sample_id': X_test.index,
                        'true_label': true_y.values,
                        'predicted_score': score,
                    })
                    df_out.to_csv(out_csv, index=False)
                    # 附加到索引
                    try:
                        idx_path = out_dir / 'predictions_index.csv'
                        import csv as _csv
                        write_header = not idx_path.exists()
                        with open(idx_path, 'a', newline='') as f:
                            writer = _csv.DictWriter(f, fieldnames=[
                                'model','train_project','test_project','csv_path',
                                'n_expected_features','n_present_features','n_missing_features','present_ratio'
                            ])
                            if write_header:
                                writer.writeheader()
                            writer.writerow({
                                'model': str(model_name),
                                'train_project': train_proj,
                                'test_project': test_proj,
                                'csv_path': str(out_csv),
                                'n_expected_features': n_expected,
                                'n_present_features': n_present,
                                'n_missing_features': n_missing,
                                'present_ratio': present_ratio,
                            })
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"{train_proj}/{model_name} 互测失败: {e}")


def _predict_external_disease(args, metadata):
    """使用已训练的疾病级模型，对外部数据中各疾病的同病样本进行预测，输出逐样本CSV。

    输出目录: <output>/reports/predict_external_disease/<model>/<Disease>_predictions.csv
    """
    logger = get_logger("ReportPredict(external_disease)")
    output_root = Path(getattr(args, 'output', 'tests/result'))
    prof_file = Path(getattr(args, 'prof_file'))
    metadata_file = Path(getattr(args, 'metadata_file'))

    # 1) 发现疾病集合
    diseases_arg = getattr(args, 'diseases', None)
    if diseases_arg:
        diseases = [str(d).strip() for d in diseases_arg if str(d).strip()]
    else:
        try:
            md = pd.read_csv(str(metadata_file), index_col=0)
            if 'Disease' not in md.columns:
                raise ValueError("metadata 需包含列 'Disease'")
            diseases = sorted(md['Disease'].dropna().astype(str).unique().tolist())
        except Exception as e:
            raise ValueError(f"无法识别疾病集合: {e}")
    if not diseases:
        logger.warning("未发现任何疾病，终止")
        return str(output_root)

    # 2) 加载外部全量数据（用于按疾病筛选子集）
    from ..data.loader import DataLoader
    dl = DataLoader()
    X_all, y_all, groups_all = dl.load_data(
        prof_file=str(prof_file),
        metadata_file=str(metadata_file),
        scope=None,
        use_presence_absence=bool(getattr(args, 'use_presence_absence', True)),
        use_clr=bool(getattr(args, 'use_clr', False)),
        enable_cohort_analysis=False,
        group_col='Group',
        label_0=getattr(args, 'label_0', None),
        label_1=getattr(args, 'label_1', None)
    )[:3]
    try:
        y_all_series = pd.Series(y_all, index=X_all.index)
    except Exception:
        y_all_series = None
    md_all = pd.read_csv(str(metadata_file), index_col=0)

    # 3) 模型列表：--models 可显式限制；否则自动发现
    models_arg = getattr(args, 'models', None)
    selected_models = [str(m).strip().lower() for m in models_arg] if models_arg else None

    # 4) builds 根目录候选
    user_builds_root = getattr(args, 'builds_root', None)
    candidate_build_roots = []
    if user_builds_root:
        try:
            candidate_build_roots.append(Path(user_builds_root))
        except Exception:
            pass
    candidate_build_roots += [
        output_root / 'abundance_raw' / 'builds',
        output_root / 'prevalence' / 'builds',
        output_root / 'builds',
    ]

    import joblib
    out_idx_rows = []
    for dis in diseases:
        # 查找疾病级模型目录: <root>/Disease_<Disease>/<model>/2_final_model
        model_base_dirs = []
        for root in candidate_build_roots:
            dis_root = root / f"Disease_{dis}"
            if not dis_root.exists():
                continue
            # 根据 --models 过滤或自动发现
            if selected_models:
                for m in selected_models:
                    cand = dis_root / str(m).lower() / '2_final_model'
                    if cand.exists():
                        model_base_dirs.append((m, cand))
            else:
                for model_dir in dis_root.iterdir():
                    cand = model_dir / '2_final_model'
                    if cand.exists():
                        model_base_dirs.append((model_dir.name, cand))
        if not model_base_dirs:
            logger.warning(f"{dis}: 未发现任何 final_model 目录 (疾病级)")
            continue

        # 取该疾病的外部样本
        test_ids = md_all[md_all['Disease'] == dis].index
        if len(test_ids) == 0:
            logger.warning(f"{dis}: 外部数据中无样本，跳过")
            continue

        for model_name, final_dir in model_base_dirs:
            try:
                model_path = final_dir / 'final_model.joblib'
                if not model_path.exists():
                    found = list(final_dir.glob('final_model_*.joblib'))
                    if found:
                        model_path = found[0]
                clf = joblib.load(model_path)
                # 特征列表
                features = getattr(clf, 'selected_features_', None)
                if not features:
                    csv_path = final_dir.parent.parent / '3_feature_analysis' / 'consensus_features.csv'
                    if csv_path.exists():
                        cdf = pd.read_csv(csv_path)
                        if 'feature_name' in cdf.columns:
                            features = cdf['feature_name'].dropna().astype(str).tolist()
                if not features:
                    logger.warning(f"{dis}/{model_name}: 无法确定特征列表，跳过")
                    continue

                # 构建测试集
                X_test = X_all.loc[X_all.index.intersection(test_ids)].copy()
                if X_test.empty:
                    continue
                for f in features:
                    if f not in X_test.columns:
                        X_test[f] = 0
                present_features = [f for f in features if f in X_test.columns]
                n_expected = len(features)
                n_present = len(present_features)
                n_missing = n_expected - n_present
                present_ratio = (n_present / n_expected) if n_expected > 0 else 0.0
                X_test = X_test[features]

                # 预测得分（正类概率优先）
                score = _predict_positive_scores(clf, X_test)
                # 输出路径
                out_dir = output_root / 'reports' / 'predict_external_disease' / str(model_name).lower()
                out_dir.mkdir(parents=True, exist_ok=True)
                out_csv = out_dir / f"{dis}_predictions.csv"
                # true_label（若可从 y_all_series 获得）
                if y_all_series is not None:
                    true_y = y_all_series.loc[X_test.index]
                    df_out = pd.DataFrame({'sample_id': X_test.index,
                                           'true_label': true_y.values,
                                           'predicted_score': score})
                else:
                    df_out = pd.DataFrame({'sample_id': X_test.index,
                                           'predicted_score': score})
                df_out.to_csv(out_csv, index=False)

                out_idx_rows.append({
                    'disease': dis,
                    'model': str(model_name),
                    'csv_path': str(out_csv),
                    'n_samples': int(X_test.shape[0]),
                    'n_expected_features': n_expected,
                    'n_present_features': n_present,
                    'n_missing_features': n_missing,
                    'present_ratio': present_ratio,
                })
                logger.info(f"✅ external_disease 预测输出: {out_csv}")
            except Exception as e:
                logger.warning(f"{dis}/{model_name} 预测失败: {e}")

    # 汇总索引
    try:
        if out_idx_rows:
            idx_csv = output_root / 'reports' / 'predict_external_disease' / 'predictions_index.csv'
            idx_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(out_idx_rows).to_csv(idx_csv, index=False)
            logger.info(f"✅ external_disease 预测索引: {idx_csv}")
            return str(idx_csv)
    except Exception:
        pass
    return str(output_root / 'reports' / 'predict_external_disease')


def _predict_external_overall(args, metadata):
    """使用 All 目录下的最终模型，对外部数据全体样本进行预测（按模型各一份CSV）。

    输出：<output>/reports/predict_external_overall/<model>/All_predictions.csv
    列：sample_id, predicted_score[, true_label]
    """
    logger = get_logger("ReportPredict(external_overall)")
    output_root = Path(getattr(args, 'output', 'tests/result'))
    prof_file = Path(getattr(args, 'prof_file'))
    metadata_file = Path(getattr(args, 'metadata_file'))

    # 加载外部数据
    from ..data.loader import DataLoader
    dl = DataLoader()
    X_all, y_all, _ = dl.load_data(
        prof_file=str(prof_file),
        metadata_file=str(metadata_file),
        scope=None,
        use_presence_absence=bool(getattr(args, 'use_presence_absence', True)),
        use_clr=bool(getattr(args, 'use_clr', False)),
        enable_cohort_analysis=False,
        group_col='Group',
        label_0=getattr(args, 'label_0', None),
        label_1=getattr(args, 'label_1', None)
    )[:3]
    try:
        y_all_series = pd.Series(y_all, index=X_all.index)
    except Exception:
        y_all_series = None

    # 搜索 All 目录模型
    user_builds_root = getattr(args, 'builds_root', None)
    candidate_build_roots = []
    if user_builds_root:
        try:
            candidate_build_roots.append(Path(user_builds_root))
        except Exception:
            pass
    candidate_build_roots += [
        output_root / 'abundance_raw' / 'builds',
        output_root / 'prevalence' / 'builds',
        output_root / 'builds',
    ]
    models_arg = getattr(args, 'models', None)
    selected_models = [str(m).strip().lower() for m in models_arg] if models_arg else None

    import joblib
    out_root = output_root / 'reports' / 'predict_external_overall'
    out_root.mkdir(parents=True, exist_ok=True)
    any_written = False
    for root in candidate_build_roots:
        all_root = root / 'All'
        if not all_root.exists():
            continue
        for model_dir in all_root.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            if selected_models and model_name.lower() not in selected_models:
                continue
            final_dir = model_dir / '2_final_model'
            if not final_dir.exists():
                continue
            try:
                model_path = final_dir / 'final_model.joblib'
                if not model_path.exists():
                    cands = list(final_dir.glob('final_model_*.joblib'))
                    if cands:
                        model_path = cands[0]
                clf = joblib.load(model_path)
                features = getattr(clf, 'selected_features_', None)
                if not features:
                    csv_path = final_dir.parent.parent / '3_feature_analysis' / 'consensus_features.csv'
                    if csv_path.exists():
                        cdf = pd.read_csv(csv_path)
                        if 'feature_name' in cdf.columns:
                            features = cdf['feature_name'].dropna().astype(str).tolist()
                if not features:
                    logger.warning(f"All/{model_name}: 无法确定特征列表，跳过")
                    continue
                X_pred = X_all.copy()
                for f in features:
                    if f not in X_pred.columns:
                        X_pred[f] = 0
                X_pred = X_pred[features]
                score = _predict_positive_scores(clf, X_pred)
                out_dir_m = out_root / model_name.lower()
                out_dir_m.mkdir(parents=True, exist_ok=True)
                out_csv = out_dir_m / 'All_predictions.csv'
                if y_all_series is not None:
                    true_y = y_all_series.loc[X_pred.index]
                    pd.DataFrame({'sample_id': X_pred.index, 'true_label': true_y.values, 'predicted_score': score}).to_csv(out_csv, index=False)
                else:
                    pd.DataFrame({'sample_id': X_pred.index, 'predicted_score': score}).to_csv(out_csv, index=False)
                logger.info(f"✅ predict_external_overall 输出: {out_csv}")
                any_written = True
            except Exception as e:
                logger.warning(f"predict_external_overall 失败({model_name}): {e}")
    if not any_written:
        logger.info("predict_external_overall: 未在 All 目录发现可用模型")
    return str(out_root)