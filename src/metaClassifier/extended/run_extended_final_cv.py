#!/usr/bin/env python3
"""
从已完成的 build 输出目录中读取 4_reproducibility/final_run.yaml，
复刻第二阶段的“参数调优CV”，导出 OOF 概率与折级指标（补救方案）。

用法示例：
python -m metaClassifier.extended.run_extended_final_cv \
  --build_dir /abs/path/to/.../builds/Disease_ASD/lasso \
  --prof_file /abs/path/to/prof.csv \
  --metadata_file /abs/path/to/metadata.csv \
  --cpu 4
"""

import argparse
from pathlib import Path
import yaml
import json
import sys

import pandas as pd

from metaClassifier.data.loader import DataLoader
from metaClassifier.extended.final_cv_evaluator import run_extended_final_cv


def _read_final_yaml(final_yaml: Path) -> dict:
    if not final_yaml.exists():
        raise FileNotFoundError(f"final_run.yaml not found: {final_yaml}")
    with open(final_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def _extract_config(d: dict) -> dict:
    # 期望 final_run.yaml 有 'config' 字段
    cfg = d.get('config', {}) if isinstance(d, dict) else {}
    # 兼容性：有些信息可能在根层
    merged = {}
    if isinstance(d, dict):
        merged.update({k: v for k, v in d.items() if k != 'config'})
    if isinstance(cfg, dict):
        merged.update(cfg)
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run extended final-stage CV export from a build directory")
    p.add_argument('--build_dir', required=True, help='已存在的某次 build 根目录，如 .../builds/Disease_ASD/lasso')
    p.add_argument('--prof_file', required=True, help='数据 prof 文件（绝对路径）')
    p.add_argument('--metadata_file', required=True, help='数据 metadata 文件（绝对路径）')
    p.add_argument('--cpu', type=int, default=4, help='并行数，默认4')
    # 可选覆盖（若未提供则由 final_run.yaml 推断/默认）
    p.add_argument('--final_cv_folds', type=int, default=None)
    p.add_argument('--final_search_method', type=str, default=None, choices=['grid', 'random', 'bayes'])
    p.add_argument('--scope', type=str, default=None, help='覆盖scope，如 "Disease=ASD"')
    p.add_argument('--use_presence_absence', type=str, choices=['true','false'], default=None, help='覆盖presence/absence设置')
    return p.parse_args()


def main():
    args = parse_args()
    build_dir = Path(args.build_dir).resolve()
    final_yaml = build_dir / '4_reproducibility' / 'final_run.yaml'
    consensus_json = build_dir / '2_final_model' / 'consensus_features.json'

    data = _read_final_yaml(final_yaml)
    cfg = _extract_config(data)

    # 提取模型名：优先从目录名；如需可从yaml兼容
    model_name = build_dir.name.lower()
    # 提取CV与搜索方法
    final_cv_folds = args.final_cv_folds if args.final_cv_folds is not None else cfg.get('final_cv_folds', cfg.get('inner_folds', 5))
    final_search_method = args.final_search_method if args.final_search_method is not None else cfg.get('final_search_method', 'grid')

    # 数据加载相关参数（与build保持一致）
    use_presence_absence = cfg.get('use_presence_absence', None)
    use_clr = cfg.get('use_clr', False)
    enable_cohort_analysis = cfg.get('enable_cohort_analysis', cfg.get('cohort_analysis', False))
    cohort_column = cfg.get('cohort_column', None)
    scope = args.scope if args.scope is not None else cfg.get('scope', None)

    # 若未在YAML中记录 presence/absence，则基于路径推断（abundance_raw -> False）
    if use_presence_absence is None:
        use_presence_absence = ('abundance_raw' not in str(build_dir))
    # 显式覆盖
    if args.use_presence_absence is not None:
        use_presence_absence = (args.use_presence_absence.lower() == 'true')
    label_0 = cfg.get('label_0', None)
    label_1 = cfg.get('label_1', None)

    # 加载数据
    dl = DataLoader()
    X, y, groups, original_features, constant_removed_features = dl.load_data(
        prof_file=args.prof_file,
        metadata_file=args.metadata_file,
        scope=scope,
        use_presence_absence=use_presence_absence,
        use_clr=use_clr,
        enable_cohort_analysis=enable_cohort_analysis,
        cohort_column=cohort_column,
        group_col="Group",
        label_0=label_0,
        label_1=label_1,
    )

    if not consensus_json.exists():
        raise FileNotFoundError(f"consensus_features.json not found: {consensus_json}")

    # 执行扩展CV导出
    results = run_extended_final_cv(
        X=X,
        y=y,
        output_root=str(build_dir),
        model_name=model_name,
        consensus_features_path=str(consensus_json),
        final_cv_folds=int(final_cv_folds),
        final_search_method=str(final_search_method),
        n_jobs=int(args.cpu),
        model_alias_for_output=model_name,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()


