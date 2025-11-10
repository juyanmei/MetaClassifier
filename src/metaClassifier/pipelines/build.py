#!/usr/bin/env python3
"""
构建流水线 for metaClassifier v1.0.

实现两阶段架构：
1. 第一阶段：嵌套CV评估（无偏性能估计 + 共识特征选择）
2. 第二阶段：最终模型训练（使用共识特征集 + 超参数调优）
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from metaClassifier.data.loader import DataLoader
from metaClassifier.core.nested_cv_classifier import create_nested_cv_classifier
from metaClassifier.core.final_model_trainer import FinalModelTrainer
from metaClassifier.core.base import CVStrategy
from metaClassifier.config import FeatureSelectionConfig
from metaClassifier.core.base import ModelConfig, CVConfig, AdaptiveFilterConfig, ExperimentConfig
from metaClassifier.utils.logger import get_logger


def handle_build(args: argparse.Namespace) -> None:
    """处理build命令：准备输出目录，加载/过滤数据，运行训练流水线。"""
    logger = get_logger("BuildPipeline")
    logger.info("开始构建流水线...")
    
    # 0. 验证参数
    _validate_args(args)
    
    # 1. 解析范围过滤器
    scope_name = 'All'
    if getattr(args, 'scope', None):
        try:
            scope_name = _parse_scope_string(args.scope)
        except Exception as e:
            logger.warning(f"解析范围过滤器失败: {e}，使用默认值 'All'")
            scope_name = 'All'
    
    # 2. 确定模型标签
    model_tag = getattr(args, 'model_name', 'lasso')
    
    # 3. 设置输出路径
    output_dir = _setup_output_directory(args, scope_name, model_tag)
    logger.info(f"输出目录: {output_dir}")
    
    # 4. 动态设置TeeLogger的日志文件
    import sys
    log_file_path = output_dir / "4_reproducibility" / "run.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, 'set_log_file') and (not hasattr(sys.stdout, 'log_file') or sys.stdout.log_file is None):
        sys.stdout.set_log_file(str(log_file_path))
        logger.info(f"日志记录已启用: {log_file_path}")
    
    # 5. 加载和准备数据
    logger.info("加载和准备数据...")
    data_loader = DataLoader()
    X, y, groups, original_features, constant_removed_features = data_loader.load_data(
        prof_file=args.prof_file,
        metadata_file=args.metadata_file,
        scope=args.scope,
        use_presence_absence=args.use_presence_absence,
        use_clr=args.use_clr,
        enable_cohort_analysis=args.enable_cohort_analysis,
        cohort_column=args.cohort_column,
        group_col="Group",  # 固定使用Group列作为标签列
        label_0=getattr(args, 'label_0', None),  # 指定标签0对应的值
        label_1=getattr(args, 'label_1', None)   # 指定标签1对应的值
    )
    
    logger.info(f"数据加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    logger.info(f"原始特征数: {len(original_features)}, 常量特征移除数: {len(constant_removed_features)}")
    
    # 5. 创建实验配置
    config = _create_experiment_config(args)
    
    # 6. 第一阶段：嵌套CV评估
    logger.info("=" * 60)
    logger.info("第一阶段：嵌套CV评估")
    logger.info("=" * 60)
    
    # 传递标签映射信息到嵌套CV评估器
    if hasattr(data_loader, 'label_mapping_'):
        logger.info(f"使用标签映射: {data_loader.label_mapping_}")
    
    # 根据重复次数选择CV策略
    if args.outer_cv_repeats > 1:
        cv_strategy = CVStrategy.REPEATED_KFOLD if args.outer_cv_strategy == 'kfold' else CVStrategy.LOCO
    else:
        cv_strategy = CVStrategy.STRATIFIED_KFOLD if args.outer_cv_strategy == 'kfold' else CVStrategy.LOCO
    
    nested_cv_classifier = create_nested_cv_classifier(
        model_name=args.model_name,
        cv_strategy=cv_strategy,
        outer_folds=args.outer_cv_folds,
        inner_folds=args.inner_cv_folds,
        n_repeats=args.outer_cv_repeats,
        enable_adaptive_filtering=args.enable_adaptive_filtering,
        output_dir=str(output_dir),
        verbose=True,
        search_method=getattr(args, 'search_method', 'grid'),
        n_jobs=getattr(args, 'cpu', 4)
    )
    
    # 传递标签映射信息到嵌套CV评估器
    if hasattr(data_loader, 'label_mapping_'):
        nested_cv_classifier.label_mapping_ = data_loader.label_mapping_
    
    # 运行嵌套CV评估
    nested_cv_results = nested_cv_classifier.evaluate(X, y, cohort_info=groups, 
                                                      original_features=original_features,
                                                      constant_removed_features=constant_removed_features)
    
    # 7. 第二阶段：最终模型训练（如果未跳过）
    if not getattr(args, 'skip_final_model', False):
        logger.info("=" * 60)
        logger.info("第二阶段：最终模型训练")
        logger.info("=" * 60)
        
        # 将输出目录写入配置，供最终阶段写入分析文件使用
        config.output_dir = str(output_dir)
        final_trainer = FinalModelTrainer(config)
        final_trainer.set_consensus_features(nested_cv_classifier.consensus_features_)
        final_trainer.set_performance_metrics(nested_cv_classifier.performance_metrics_)
        
        # 传递自适应参数信息（用于可重现性）
        if hasattr(nested_cv_classifier, 'adaptive_param_info_'):
            final_trainer.adaptive_param_info_ = nested_cv_classifier.adaptive_param_info_
        if hasattr(nested_cv_classifier, 'class_balance_info_'):
            final_trainer.class_balance_info_ = nested_cv_classifier.class_balance_info_
        
        # 传递外层CV特有信息（用于可重现性）
        if hasattr(nested_cv_classifier, 'performance_metrics_'):
            final_trainer.performance_metrics_ = nested_cv_classifier.performance_metrics_
        if hasattr(nested_cv_classifier, 'selection_stats_'):
            final_trainer.selection_stats_ = nested_cv_classifier.selection_stats_
        if hasattr(nested_cv_classifier, 'outer_fold_summary_'):
            final_trainer.outer_fold_summary_ = nested_cv_classifier.outer_fold_summary_
        
        # 传递标签映射信息
        if hasattr(data_loader, 'label_mapping_'):
            final_trainer.label_mapping_ = data_loader.label_mapping_
        
        # 训练最终模型
        final_model_results = final_trainer.train_final_models(
            X, y, cohort_info=groups,
            final_cv_folds=getattr(args, 'final_cv_folds', 5),
            final_search_method=getattr(args, 'final_search_method', None) or getattr(args, 'search_method', 'grid'),
            cpu=getattr(args, 'cpu', 4)
        )
        
        # 保存最终模型
        final_trainer.save_final_models(output_dir)
        logger.info("最终模型训练完成")
    else:
        logger.info("跳过最终模型训练")
    
    logger.info("构建流水线完成！")


def _parse_scope_string(scope: str) -> str:
    """解析范围字符串，返回安全的范围名称。"""
    if not scope or '=' not in scope:
        return 'All'
    
    column, value = scope.split('=', 1)
    # 清理值，移除特殊字符
    safe_value = ''.join(c for c in value if c.isalnum() or c in '_-')
    return f"{column}_{safe_value}"


def _setup_output_directory(args: argparse.Namespace, scope_name: str, model_tag: str) -> Path:
    """设置输出目录。"""
    # 生成数据处理标签
    data_type_tag = ""
    if getattr(args, 'use_presence_absence', True):
        data_type_tag = "prevalence"
    else:
        if getattr(args, 'use_clr', False):
            data_type_tag = "abundance_clr"
        else:
            data_type_tag = "abundance_raw"
    
    if args.output:
        # 用户指定了根目录
        output_dir = Path(args.output) / data_type_tag / "builds" / scope_name / model_tag
    else:
        # 使用默认的results根目录
        output_dir = Path("results") / data_type_tag / "builds" / scope_name / model_tag
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _create_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """从命令行参数创建实验配置。"""
    # 模型配置
    model_config = ModelConfig(
        name=args.model_name,
        task_type='classification',
        hyperparameters={}
    )
    
    # CV配置 - 支持重复CV
    outer_cv_strategy = getattr(args, 'outer_cv_strategy', 'kfold')
    if args.outer_cv_repeats > 1:
        cv_strategy = CVStrategy.REPEATED_KFOLD if outer_cv_strategy == 'kfold' else CVStrategy.LOCO
    else:
        cv_strategy = CVStrategy.STRATIFIED_KFOLD if outer_cv_strategy == 'kfold' else CVStrategy.LOCO
    
    cv_config = CVConfig(
        strategy=cv_strategy,
        outer_folds=args.outer_cv_folds,
        inner_folds=args.inner_cv_folds,
        n_repeats=args.outer_cv_repeats,
        random_state=42,
        n_jobs=args.cpu
    )
    
    # 自适应过滤配置 (使用内部优化的默认参数)
    adaptive_filter_config = AdaptiveFilterConfig(
        enabled=args.enable_adaptive_filtering,
        min_q=getattr(args, 'min_q', 0.5),
        max_q=getattr(args, 'max_q', 0.95),
        r_mid=getattr(args, 'r_mid', 1.0),
        steepness=getattr(args, 'steepness', 2.0)
    )
    
    # 特征选择配置
    feature_selection_config = FeatureSelectionConfig(
        enabled=args.feature_selection,
        threshold=args.feature_threshold,
        search_method=args.search_method
    )
    
    # 实验配置
    experiment_config = ExperimentConfig(
        model=model_config,
        cv=cv_config,
        adaptive_filter=adaptive_filter_config,
        feature_selection=feature_selection_config,  # 新增
        data_paths={
            'prof_file': args.prof_file,
            'metadata_file': args.metadata_file
        },
        output_dir=None,  # 将在运行时设置
        verbose=True
    )
    
    # 记录未使用的参数（用于日志）
    logger = get_logger("BuildPipeline")
    logger.info(f"特征选择: {args.feature_selection}")
    # 移除未使用的特征策略参数输出
    logger.info(f"超参数搜索方法: {args.search_method}")
    logger.info(f"外层CV重复次数: {args.outer_cv_repeats}")
    logger.info(f"队列分析: {args.enable_cohort_analysis}")
    logger.info(f"队列列名: {args.cohort_column}")
    
    return experiment_config


def _validate_args(args: argparse.Namespace) -> None:
    """验证参数的有效性。"""
    from pathlib import Path
    
    # 验证文件存在
    if not Path(args.prof_file).exists():
        raise FileNotFoundError(f"Profile文件不存在: {args.prof_file}")
    if not Path(args.metadata_file).exists():
        raise FileNotFoundError(f"Metadata文件不存在: {args.metadata_file}")
    
    # 验证参数组合
    if args.use_clr and args.use_presence_absence:
        raise ValueError("CLR变换只能用于相对丰度数据，不能与有无数据同时使用")
    
    # 验证CV参数
    if args.outer_cv_folds < 2:
        raise ValueError("外层CV折数必须大于等于2")
    if args.inner_cv_folds < 2:
        raise ValueError("内层CV折数必须大于等于2")
    if args.outer_cv_repeats < 1:
        raise ValueError("外层CV重复次数必须大于等于1")
    
    # 验证特征选择参数
    if not 0 <= args.feature_threshold <= 1:
        raise ValueError("特征阈值必须在0-1之间")
    
    # 自适应过滤参数已内部优化，无需用户验证
    
    # 验证系统参数
    if args.cpu < 1:
        raise ValueError("CPU核心数必须大于0")
    # 随机种子内部固定为42，无需校验
