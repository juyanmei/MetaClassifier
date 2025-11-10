"""
参数解析器 for metaClassifier v1.0.

借鉴原始项目的参数结构，支持build和report两种模式。
"""

import argparse
from typing import List


def str2bool(v):
    """将字符串转换为布尔值。"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def comma_separated_items(value: str) -> List[str]:
    """解析逗号分隔的字符串为列表。"""
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def create_argument_parser() -> argparse.ArgumentParser:
    """创建和配置CLI解析器，支持build和report子命令。"""
    parser = argparse.ArgumentParser(
        description="metaClassifier v1.0 - 宏基因组分类模型构建终极指南",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    def add_common_options(p):
        """添加通用参数选项。"""
        # 数据文件参数
        p.add_argument('--prof_file', type=str, required=True,
                       help="Profile数据文件路径 (行=样本, 列=物种)")
        p.add_argument('--metadata_file', type=str, required=True,
                       help="元数据文件路径")
        p.add_argument('--output', type=str, required=False, default=None,
                       help="结果输出目录 (未指定时自动生成)")
        
        # 模型参数
        p.add_argument('--model_name', type=str, required=False, default='lasso',
                       choices=['lasso', 'elasticnet', 'logistic', 'randomforest', 'catboost', 'neuralnetwork'],
                       help="要使用的模型名称")
        
        # 交叉验证参数
        p.add_argument('--outer_cv_strategy', type=str, required=False, default='kfold',
                       choices=['kfold', 'loco'],
                       help="外层CV策略: kfold (K折交叉验证) 或 loco (Leave-One-Cohort-Out)")
        p.add_argument('--outer_cv_folds', type=int, required=False, default=5,
                       help="外层CV折数")
        p.add_argument('--inner_cv_folds', type=int, required=False, default=3,
                       help="内层CV折数")
        p.add_argument('--outer_cv_repeats', type=int, required=False, default=1,
                       help="外层CV重复次数")
        
        # 数据处理参数
        p.add_argument('--use_presence_absence', type=str2bool, required=False, default=True,
                       help="使用有无数据 (默认: True)")
        p.add_argument('--use_clr', type=str2bool, required=False, default=False,
                       help="对相对丰度数据应用CLR变换 (仅在use_presence_absence=False时有效)")
        
        # 特征选择参数（保留必要项）
        p.add_argument('--feature_selection', type=str2bool, required=False, default=True,
                       help="启用特征选择 (默认: True)")
        p.add_argument('--feature_threshold', type=float, required=False, default=0.5,
                       help="一致特征选择的频率阈值 (0-1)")
        
        # 超参数调优参数
        p.add_argument('--search_method', type=str, required=False, default='grid',
                       choices=['random', 'grid', 'bayes'],
                       help="超参数搜索方法（grid/random/bayes）")

        # 最终模型阶段的CV与搜索
        p.add_argument('--final_cv_folds', type=int, required=False, default=5,
                       help='最终模型阶段用于选参的CV折数')
        p.add_argument('--final_search_method', type=str, required=False, default=None,
                       choices=['random', 'grid', 'bayes'],
                       help='最终模型阶段的超参数搜索方法（grid/random/bayes；默认沿用 --search_method）')
        
        # 自适应方差过滤参数
        p.add_argument('--enable_adaptive_filtering', type=str2bool, required=False, default=True,
                       help="启用自适应方差过滤 (默认: True)")
        
        # 系统参数
        p.add_argument('--cpu', type=int, required=False, default=4,
                       help="CPU核心数")
        
        # 配置文件
        p.add_argument('--config', type=str, required=False, default=None,
                       help="YAML配置文件路径 (可选，覆盖默认值)")
        
        # 队列分析参数
        p.add_argument('--enable_cohort_analysis', type=str2bool, required=False, default=False,
                       help="启用队列分析用于LOCO验证")
        p.add_argument('--cohort_column', type=str, required=False, default='Project',
                       choices=['Project', 'Disease'],
                       help="队列信息列名 (默认: Project)")
        
        # 标签映射参数
        p.add_argument('--label_0', type=str, required=False, default=None,
                       help="指定标签0对应的Group值 (例如: Health, Control)")
        p.add_argument('--label_1', type=str, required=False, default=None,
                       help="指定标签1对应的Group值 (例如: Disease, Case)")
    
    # build子命令
    build_p = subparsers.add_parser('build', help='构建和训练模型 (包含嵌套CV评估和最终模型训练)')
    add_common_options(build_p)
    build_p.add_argument('--scope', type=str, required=False,
                         help="范围过滤器，格式为'列名=值'，应用于训练前的元数据过滤 (例如: Project=ProjectA)")
    build_p.add_argument('--skip_final_model', type=str2bool, required=False, default=False,
                         help="跳过最终模型训练，仅进行嵌套CV评估")
    
    # report子命令 - 使用复杂设计但简化help输出
    report_p = subparsers.add_parser('report', help='生成分析报告')
    
    # 添加所有训练参数（内部使用）
    add_common_options(report_p)
    
    # 报告生成参数
    report_p.add_argument('--scenario', type=str, required=True,
                          choices=['within_disease', 'between_project', 'between_disease', 'overall', 'models', 'predict', 'predict_external_disease', 'predict_external_overall'],
                          help='预定义分析场景')
    report_p.add_argument('--disease', type=str, required=False,
                          help='疾病名称 (within_disease场景必需)')
    report_p.add_argument('--metric', type=str, required=False, default='auc',
                          choices=['auc', 'accuracy', 'f1', 'precision', 'recall'],
                          help='用于填充交叉性能矩阵的指标')
    report_p.add_argument('--scope', type=str, required=False,
                          help="范围过滤器，格式为'列名=值' (例如: Project=ProjectA)")
    report_p.add_argument('--models', type=comma_separated_items, required=False,
                          help='models场景中要比较的模型列表 (逗号分隔)')
    report_p.add_argument('--diseases', type=comma_separated_items, required=False,
                          help='predict_external_disease场景中要预测的疾病列表 (逗号分隔；缺省自动从metadata中发现)')
    report_p.add_argument('--emit_predictions', type=str2bool, required=False, default=False,
                          help='在report中额外输出逐样本预测CSV（使用已训练final_model）')
    report_p.add_argument('--order', type=str, required=False, default=None,
                          help='顺序文件路径（CSV），如包含列 ProjectOrder, ProjectID，用于热图行列顺序')
    # 可选的builds路径覆写
    report_p.add_argument('--builds_root', type=str, required=False, default=None,
                          help='显式指定 builds 根路径（优先使用该路径）')
    
    return parser


def create_custom_report_parser() -> argparse.ArgumentParser:
    """创建自定义的report解析器，只显示report相关参数"""
    parser = argparse.ArgumentParser(
        description="metaClassifier v1.0 - 生成分析报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 只添加report相关的参数（补充关键数据处理与排序参数）
    parser.add_argument('--prof_file', type=str, required=True,
                       help="Profile数据文件路径 (行=样本, 列=物种)")
    parser.add_argument('--metadata_file', type=str, required=True,
                       help="元数据文件路径")
    parser.add_argument('--output', type=str, required=False, default='tests/result',
                       help="结果输出目录")
    
    # 报告生成参数
    parser.add_argument('--scenario', type=str, required=True,
                       choices=['within_disease', 'between_project', 'between_disease', 'overall', 'models', 'predict_external_disease', 'predict_external_overall'],
                       help='预定义分析场景')
    parser.add_argument('--disease', type=str, required=False,
                       help='疾病名称 (within_disease场景必需)')
    parser.add_argument('--metric', type=str, required=False, default='auc',
                       choices=['auc', 'accuracy', 'f1', 'precision', 'recall'],
                       help='用于填充交叉性能矩阵的指标')
    parser.add_argument('--scope', type=str, required=False,
                       help="范围过滤器，格式为'列名=值' (例如: Project=ProjectA)")
    parser.add_argument('--models', type=comma_separated_items, required=False,
                       help='models场景比较的模型列表 / predict_external_disease场景用于筛选模型 (逗号分隔)')
    parser.add_argument('--diseases', type=comma_separated_items, required=False,
                       help='predict_external_disease场景中要预测的疾病列表 (逗号分隔；缺省自动从metadata发现)')
    parser.add_argument('--emit_predictions', type=str2bool, required=False, default=False,
                       help='在report中额外输出逐样本预测CSV（使用已训练final_model）')
    # 数据处理与排序可见
    parser.add_argument('--use_presence_absence', type=str2bool, required=False, default=True,
                       help="使用有无数据 (默认: True)")
    parser.add_argument('--use_clr', type=str2bool, required=False, default=False,
                       help="对相对丰度数据应用CLR变换 (仅在use_presence_absence=False时有效)")
    parser.add_argument('--order', type=str, required=False, default=None,
                       help='顺序文件路径（CSV），如包含列 ProjectOrder, ProjectID，用于热图行列顺序')
    parser.add_argument('--builds_root', type=str, required=False, default=None,
                       help='显式指定 builds 根路径（优先使用该路径）')
    
    return parser


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    import sys
    
    # 检查是否是report --help请求
    if len(sys.argv) >= 2 and sys.argv[1] == 'report' and '--help' in sys.argv:
        # 使用自定义的report解析器显示简化的help
        custom_parser = create_custom_report_parser()
        custom_parser.parse_args()
        sys.exit(0)
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 统一参数映射：保持内部使用简洁的参数名
    # 注意：不再进行model_name到model的映射，直接使用model_name
    if hasattr(args, 'outer_cv_repeats'):
        setattr(args, 'n_repeats', getattr(args, 'outer_cv_repeats'))
    
    # YAML配置合并（简化版本：配置文件覆盖默认值）
    cfg_path = getattr(args, 'config', None) if hasattr(args, 'config') else None
    if cfg_path:
        setattr(args, 'config_file', cfg_path)
        try:
            import yaml
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            # 将YAML配置合并到args中
            for key, value in cfg.items():
                if not hasattr(args, key):
                    setattr(args, key, value)
        except Exception as e:
            raise ValueError(f"Failed to load YAML config: {e}")
    
    return args
