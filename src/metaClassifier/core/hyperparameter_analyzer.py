"""
超参数选择分析器 for metaClassifier v1.0.

提供超参数选择的统计分析和可视化功能。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from ..utils.logger import get_logger


class HyperparameterAnalyzer:
    """
    超参数选择分析器。
    
    提供超参数选择的统计分析和可视化功能：
    1. 超参数重要性分析
    2. 超参数分布可视化
    3. 超参数选择历史分析
    4. 超参数稳定性分析
    """
    
    def __init__(self, output_dir: str = "./hyperparameter_analysis"):
        """
        初始化超参数分析器。
        
        Args:
            output_dir: 分析结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("HyperparameterAnalyzer")
        
        # 存储分析结果
        self.analysis_results_ = {}
        self.hyperparameter_history_ = []
        
    def analyze_hyperparameter_selection(
        self, 
        model_name: str,
        cv_results: Dict[str, Any],
        search_method: str = "grid"
    ) -> Dict[str, Any]:
        """
        分析超参数选择结果。
        
        Args:
            model_name: 模型名称
            cv_results: 交叉验证结果
            search_method: 搜索方法
            
        Returns:
            分析结果字典
        """
        self.logger.info(f"分析 {model_name} 的超参数选择结果...")
        
        analysis = {
            'model_name': model_name,
            'search_method': search_method,
            'best_params': cv_results.get('best_params_', {}),
            'best_score': cv_results.get('best_score_', 0.0),
            'param_importance': self._calculate_param_importance(cv_results),
            'param_distribution': self._analyze_param_distribution(cv_results),
            'score_distribution': self._analyze_score_distribution(cv_results),
            'convergence_analysis': self._analyze_convergence(cv_results)
        }
        
        # 保存分析结果
        self.analysis_results_[model_name] = analysis
        
        # 生成可视化
        self._generate_visualizations(model_name, analysis)
        
        # 保存详细结果
        self._save_analysis_results(model_name, analysis)
        
        self.logger.info(f"{model_name} 超参数分析完成")
        return analysis
    
    def _calculate_param_importance(self, cv_results: Dict[str, Any]) -> Dict[str, float]:
        """计算超参数重要性。"""
        param_importance = {}
        
        if 'cv_results_' in cv_results and cv_results['cv_results_'] is not None:
            # 使用sklearn的cv_results_计算参数重要性
            results = cv_results['cv_results_']
            
            for param_name in results['params'][0].keys():
                param_values = [params[param_name] for params in results['params']]
                param_scores = results['mean_test_score']
                
                # 计算参数值与得分的相关性
                if len(set(param_values)) > 1:  # 确保参数值有变化
                    correlation = np.corrcoef(param_values, param_scores)[0, 1]
                    param_importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return param_importance
    
    def _analyze_param_distribution(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析超参数分布。"""
        param_distribution = {}
        
        if 'cv_results_' in cv_results and cv_results['cv_results_'] is not None:
            results = cv_results['cv_results_']
            
            for param_name in results['params'][0].keys():
                param_values = [params[param_name] for params in results['params']]
                
                param_distribution[param_name] = {
                    'unique_values': list(set(param_values)),
                    'value_counts': pd.Series(param_values).value_counts().to_dict(),
                    'mean': np.mean(param_values) if all(isinstance(v, (int, float)) for v in param_values) else None,
                    'std': np.std(param_values) if all(isinstance(v, (int, float)) for v in param_values) else None
                }
        
        return param_distribution
    
    def _analyze_score_distribution(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析得分分布。"""
        if 'cv_results_' in cv_results and cv_results['cv_results_'] is not None:
            results = cv_results['cv_results_']
            scores = results['mean_test_score']
            
            return {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75)
            }
        
        return {}
    
    def _analyze_convergence(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析收敛性（仅适用于贝叶斯优化）。"""
        convergence_analysis = {}
        
        if 'cv_results_' in cv_results and cv_results['cv_results_'] is not None:
            results = cv_results['cv_results_']
            scores = results['mean_test_score']
            
            # 计算收敛指标
            convergence_analysis = {
                'score_improvement': scores[-1] - scores[0] if len(scores) > 1 else 0,
                'score_variance': np.var(scores),
                'convergence_rate': self._calculate_convergence_rate(scores)
            }
        
        return convergence_analysis
    
    def _calculate_convergence_rate(self, scores: List[float]) -> float:
        """计算收敛率。"""
        if len(scores) < 2:
            return 0.0
        
        # 计算最后10%的搜索中得分改善程度
        last_portion = max(1, len(scores) // 10)
        recent_scores = scores[-last_portion:]
        
        if len(recent_scores) < 2:
            return 0.0
        
        improvement = recent_scores[-1] - recent_scores[0]
        return improvement / len(recent_scores)
    
    def _generate_visualizations(self, model_name: str, analysis: Dict[str, Any]):
        """生成可视化图表。"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 超参数重要性图
            self._plot_param_importance(model_name, analysis['param_importance'])
            
            # 2. 得分分布图
            self._plot_score_distribution(model_name, analysis['score_distribution'])
            
            # 3. 超参数分布图
            self._plot_param_distribution(model_name, analysis['param_distribution'])
            
        except Exception as e:
            self.logger.warning(f"生成可视化图表失败: {e}")
    
    def _plot_param_importance(self, model_name: str, param_importance: Dict[str, float]):
        """绘制超参数重要性图。"""
        if not param_importance:
            return
        
        plt.figure(figsize=(10, 6))
        params = list(param_importance.keys())
        importance = list(param_importance.values())
        
        bars = plt.bar(params, importance, color='skyblue', alpha=0.7)
        plt.title(f'{model_name} 超参数重要性分析')
        plt.xlabel('超参数')
        plt.ylabel('重要性得分')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, imp in zip(bars, importance):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{imp:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_param_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distribution(self, model_name: str, score_distribution: Dict[str, Any]):
        """绘制得分分布图。"""
        if not score_distribution:
            return
        
        # 这里需要实际的得分数据，暂时跳过
        pass
    
    def _plot_param_distribution(self, model_name: str, param_distribution: Dict[str, Any]):
        """绘制超参数分布图。"""
        if not param_distribution:
            return
        
        # 选择数值型参数进行分布分析
        numeric_params = {k: v for k, v in param_distribution.items() 
                         if v['mean'] is not None}
        
        if not numeric_params:
            return
        
        n_params = len(numeric_params)
        if n_params == 0:
            return
        
        fig, axes = plt.subplots(1, min(n_params, 3), figsize=(15, 5))
        if n_params == 1:
            axes = [axes]
        
        for i, (param_name, param_info) in enumerate(list(numeric_params.items())[:3]):
            if i >= len(axes):
                break
            
            # 这里需要实际的参数值数据，暂时跳过
            pass
    
    def _save_analysis_results(self, model_name: str, analysis: Dict[str, Any]):
        """保存分析结果。"""
        # 保存JSON格式的详细结果
        json_path = self.output_dir / f'{model_name}_hyperparameter_analysis.json'
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            else:
                return convert_numpy(d)
        
        analysis_clean = recursive_convert(analysis)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_clean, f, indent=2, ensure_ascii=False)
    
    def generate_summary_report(self) -> str:
        """生成总结报告。"""
        if not self.analysis_results_:
            return "没有可用的分析结果"
        
        report = []
        report.append("# 超参数选择分析报告")
        report.append("=" * 50)
        
        for model_name, analysis in self.analysis_results_.items():
            report.append(f"\n## {model_name}")
            report.append(f"- 搜索方法: {analysis['search_method']}")
            report.append(f"- 最佳得分: {analysis['best_score']:.4f}")
            report.append(f"- 最佳参数: {analysis['best_params']}")
            
            if analysis['param_importance']:
                report.append("- 参数重要性排序:")
                sorted_params = sorted(analysis['param_importance'].items(), 
                                     key=lambda x: x[1], reverse=True)
                for param, importance in sorted_params:
                    report.append(f"  - {param}: {importance:.3f}")
        
        return "\n".join(report)
