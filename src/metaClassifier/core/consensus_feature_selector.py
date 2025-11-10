"""
共识特征选择器 for metaClassifier v1.0.

提供多种共识特征选择策略，用于从嵌套CV结果中选择稳定的特征集。
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger


class ConsensusStrategy(Enum):
    """共识特征选择策略。"""
    MAJORITY_VOTING = "majority_voting"  # 多数投票
    WEIGHTED_VOTING = "weighted_voting"  # 加权投票
    STABILITY_SCORE = "stability_score"  # 稳定性评分
    IMPORTANCE_BASED = "importance_based"  # 基于重要性
    ADAPTIVE = "adaptive"  # 自适应策略


@dataclass
class ConsensusConfig:
    """共识特征选择配置。"""
    strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY_VOTING
    threshold: float = 0.5  # 投票阈值
    min_features: int = 10  # 最小特征数
    max_features: int = 1000  # 最大特征数
    stability_threshold: float = 0.7  # 稳定性阈值
    importance_threshold: float = 0.1  # 重要性阈值
    adaptive_ratio: float = 0.3  # 自适应比例


class ConsensusFeatureSelector:
    """
    共识特征选择器。
    
    提供多种策略来选择稳定的特征集：
    1. 多数投票：基于特征在多个折中的出现频率
    2. 加权投票：考虑特征重要性的加权投票
    3. 稳定性评分：基于特征选择的一致性
    4. 基于重要性：基于特征重要性排序
    5. 自适应策略：根据数据特征自动调整
    """
    
    def __init__(self, config: ConsensusConfig):
        """
        初始化共识特征选择器。
        
        Args:
            config: 共识特征选择配置
        """
        self.config = config
        self.logger = get_logger("ConsensusFeatureSelector")
        
        self.consensus_features_ = {}
        self.feature_scores_ = {}
        self.selection_stats_ = {}
    
    def select_consensus_features(
        self, 
        outer_fold_results: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        选择共识特征。
        
        Args:
            outer_fold_results: 外层折结果列表
            
        Returns:
            共识特征字典
        """
        self.logger.info(f"使用策略 {self.config.strategy.value} 选择共识特征")
        
        if self.config.strategy == ConsensusStrategy.MAJORITY_VOTING:
            return self._majority_voting(outer_fold_results)
        elif self.config.strategy == ConsensusStrategy.WEIGHTED_VOTING:
            return self._weighted_voting(outer_fold_results)
        elif self.config.strategy == ConsensusStrategy.STABILITY_SCORE:
            return self._stability_score(outer_fold_results)
        elif self.config.strategy == ConsensusStrategy.IMPORTANCE_BASED:
            return self._importance_based(outer_fold_results)
        elif self.config.strategy == ConsensusStrategy.ADAPTIVE:
            return self._adaptive_selection(outer_fold_results)
        else:
            raise ValueError(f"Unknown consensus strategy: {self.config.strategy}")
    
    def _majority_voting(self, outer_fold_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """多数投票策略 - 支持三层筛选。"""
        self.logger.info("使用多数投票策略选择共识特征（三层筛选）")
        
        consensus_features = {}
        
        for model_name in outer_fold_results[0]['model_results'].keys():
            # 执行三层筛选
            filtering_results = self._three_layer_filtering(outer_fold_results, model_name)
            final_features = filtering_results['repeat_features']
            consensus_features[model_name] = final_features
            
            # 记录统计信息
            self.selection_stats_[model_name] = self._get_selection_stats(outer_fold_results, model_name, final_features)
            
            # 根据模式显示不同的筛选信息
            is_loco = self._is_loco_mode(outer_fold_results)
            if is_loco:
                self.logger.info(f"{model_name}: {len(final_features)} 个共识特征（通过LOCO两层筛选）")
            else:
                self.logger.info(f"{model_name}: {len(final_features)} 个共识特征（通过三层筛选）")
        
        return consensus_features
    
    def _three_layer_filtering(self, outer_fold_results: List[Dict[str, Any]], model_name: str) -> Dict[str, List[str]]:
        """三层筛选：内层CV -> 外层CV -> 重复筛选"""
        
        # 检查是否为LOCO模式
        is_loco = self._is_loco_mode(outer_fold_results)
        
        if is_loco:
            # LOCO模式：使用两层筛选（内层CV -> 队列筛选）
            return self._loco_two_layer_filtering(outer_fold_results, model_name)
        else:
            # 重复CV模式：使用三层筛选
            return self._repeated_cv_three_layer_filtering(outer_fold_results, model_name)
    
    def _is_loco_mode(self, outer_fold_results: List[Dict[str, Any]]) -> bool:
        """检查是否为LOCO模式"""
        # 直接使用CV策略参数判断，这是最可靠的方法
        from .base import CVStrategy
        return self.config.strategy == CVStrategy.LOCO
    
    def _loco_two_layer_filtering(self, outer_fold_results: List[Dict[str, Any]], model_name: str) -> Dict[str, List[str]]:
        """LOCO模式的两层筛选：内层CV -> 队列筛选"""
        self.logger.info("LOCO模式：使用两层筛选（内层CV -> 队列筛选）")
        
        # 第一层：收集所有通过各自内层CV筛选的特征
        inner_cv_features = self._inner_cv_filtering(outer_fold_results, model_name)
        self.logger.info(f"  第一层（内层CV）筛选: {len(inner_cv_features)} 个特征")
        
        # 第二层：队列筛选（基于第一层的结果）
        cohort_features = self._cohort_filtering(outer_fold_results, model_name, inner_cv_features)
        self.logger.info(f"  第二层（队列）筛选: {len(cohort_features)} 个特征")
        
        return {
            'inner_cv_features': inner_cv_features,
            'outer_cv_features': cohort_features,  # 在LOCO中，外层CV就是队列筛选
            'repeat_features': cohort_features     # 没有重复筛选，直接使用队列筛选结果
        }
    
    def _repeated_cv_three_layer_filtering(self, outer_fold_results: List[Dict[str, Any]], model_name: str) -> Dict[str, List[str]]:
        """重复CV模式的三层筛选：内层CV -> 外层CV -> 重复筛选"""
        self.logger.info("重复CV模式：使用三层筛选（内层CV -> 外层CV -> 重复筛选）")
        
        # 第一层：收集所有通过各自内层CV筛选的特征
        inner_cv_features = self._inner_cv_filtering(outer_fold_results, model_name)
        self.logger.info(f"  第一层（内层CV）筛选: {len(inner_cv_features)} 个特征")
        
        # 第二层：外层CV筛选（基于第一层的结果）
        outer_cv_features = self._outer_cv_filtering(outer_fold_results, model_name, inner_cv_features)
        self.logger.info(f"  第二层（外层CV）筛选: {len(outer_cv_features)} 个特征")
        
        # 第三层：重复筛选（基于前两层的结果）
        repeat_features = self._repeat_filtering(outer_fold_results, model_name, outer_cv_features)
        self.logger.info(f"  第三层（重复）筛选: {len(repeat_features)} 个特征")
        
        return {
            'inner_cv_features': inner_cv_features,
            'outer_cv_features': outer_cv_features,
            'repeat_features': repeat_features
        }
    
    def _cohort_filtering(self, outer_fold_results: List[Dict[str, Any]], model_name: str, candidate_features: List[str]) -> List[str]:
        """队列筛选：基于队列稳定性筛选特征"""
        # 统计每个特征在多少个队列中被选中
        feature_cohort_counts = {}
        for fold_result in outer_fold_results:
            fold_features = fold_result['model_results'][model_name]['selected_features']
            # 只保留通过第一层筛选的特征
            filtered_features = [f for f in fold_features if f in candidate_features]
            
            # 统计特征在该队列中的出现
            for feature in candidate_features:
                if feature not in feature_cohort_counts:
                    feature_cohort_counts[feature] = 0
                if feature in filtered_features:
                    feature_cohort_counts[feature] += 1
        
        # 计算队列阈值（基于队列数）
        cohort_count = len(outer_fold_results)
        cohort_threshold = max(1, int(cohort_count * self.config.threshold))
        
        # 选择满足队列阈值的特征
        cohort_features = [
            feature for feature, count in feature_cohort_counts.items()
            if count >= cohort_threshold
        ]
        
        self.logger.info(f"    队列筛选: {cohort_count}个队列, 阈值={cohort_threshold}, 选中={len(cohort_features)}个特征")
        return cohort_features
    
    def _inner_cv_filtering(self, outer_fold_results: List[Dict[str, Any]], model_name: str) -> List[str]:
        """第一层：内层CV筛选 - 每个外层折的内层CV都需要满足阈值"""
        # 收集所有通过各自内层CV筛选的特征
        all_inner_cv_passed_features = []
        
        for fold_result in outer_fold_results:
            # 获取该外层折经过内层CV筛选后的特征
            selected_features = fold_result['model_results'][model_name]['selected_features']
            all_inner_cv_passed_features.extend(selected_features)
        
        # 统计特征在外层CV中被选中的总次数
        feature_counts = Counter(all_inner_cv_passed_features)
        
        # 计算外层CV阈值（基于外层折数）
        outer_fold_count = len(outer_fold_results)
        outer_threshold = max(1, int(outer_fold_count * self.config.threshold))
        
        # 选择满足外层CV阈值的特征
        inner_cv_features = [
            feature for feature, count in feature_counts.items()
            if count >= outer_threshold
        ]
        
        self.logger.info(f"    第一层（内层CV）: {outer_fold_count}个外层折, 阈值={outer_threshold}, 选中={len(inner_cv_features)}个特征")
        return inner_cv_features
    
    def _outer_cv_filtering(self, outer_fold_results: List[Dict[str, Any]], model_name: str, candidate_features: List[str]) -> List[str]:
        """第二层：外层CV筛选 - 基于第一层结果进行外层CV筛选"""
        # 只考虑通过第一层筛选的特征
        outer_cv_features = []
        for fold_result in outer_fold_results:
            fold_features = fold_result['model_results'][model_name]['selected_features']
            # 只保留通过第一层筛选的特征
            filtered_features = [f for f in fold_features if f in candidate_features]
            outer_cv_features.extend(filtered_features)
        
        # 计算外层CV阈值（基于外层折数）
        outer_fold_count = len(outer_fold_results)
        outer_threshold = max(1, int(outer_fold_count * self.config.threshold))
        
        # 统计特征出现次数
        feature_counts = Counter(outer_cv_features)
        
        # 选择满足外层CV阈值的特征
        outer_cv_features = [
            feature for feature, count in feature_counts.items()
            if count >= outer_threshold
        ]
        
        self.logger.info(f"    第二层（外层CV）: {outer_fold_count}折, 阈值={outer_threshold}, 选中={len(outer_cv_features)}个特征")
        return outer_cv_features
    
    def _repeat_filtering(self, outer_fold_results: List[Dict[str, Any]], model_name: str, candidate_features: List[str]) -> List[str]:
        """第三层：重复筛选 - 基于前两层结果进行重复筛选"""
        # 按重复分组统计特征出现次数
        repeat_groups = {}
        for fold_result in outer_fold_results:
            repeat_idx = fold_result.get('repeat_idx', 0)
            if repeat_idx not in repeat_groups:
                repeat_groups[repeat_idx] = []
            
            fold_features = fold_result['model_results'][model_name]['selected_features']
            # 只保留通过前两层筛选的特征
            filtered_features = [f for f in fold_features if f in candidate_features]
            repeat_groups[repeat_idx].extend(filtered_features)
        
        # 统计每个特征在多少个重复中被选中
        feature_repeat_counts = {}
        for repeat_idx, features in repeat_groups.items():
            # 对于每个候选特征，检查是否在该重复中被选中
            for feature in candidate_features:
                if feature not in feature_repeat_counts:
                    feature_repeat_counts[feature] = 0
                if feature in features:  # 只要在该重复中出现过，就记作1
                    feature_repeat_counts[feature] += 1
        
        # 计算重复阈值（基于重复次数）
        repeat_count = len(repeat_groups)
        repeat_threshold = max(1, int(repeat_count * self.config.threshold))
        
        # 选择满足重复阈值的特征
        repeat_features = [
            feature for feature, count in feature_repeat_counts.items()
            if count >= repeat_threshold
        ]
        
        self.logger.info(f"    第三层（重复）: {repeat_count}次重复, 阈值={repeat_threshold}, 选中={len(repeat_features)}个特征")
        return repeat_features
    
    def _get_selection_stats(self, outer_fold_results: List[Dict[str, Any]], model_name: str, final_features: List[str]) -> Dict[str, Any]:
        """获取选择统计信息"""
        # 统计所有特征
        all_features = []
        for fold_result in outer_fold_results:
            features = fold_result['model_results'][model_name]['selected_features']
            all_features.extend(features)
            
            feature_counts = Counter(all_features)
        
        # 获取三层筛选的详细统计
        three_layer_stats = self._get_three_layer_stats(outer_fold_results, model_name)
        
        return {
            'total_features': len(feature_counts),
            'consensus_features': len(final_features),
            'threshold': self.config.threshold,
            'feature_counts': dict(feature_counts),
            'final_features': final_features,
            'three_layer_filtering_stats': three_layer_stats
        }
    
    def _get_three_layer_stats(self, outer_fold_results: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        """获取三层筛选的详细统计信息"""
        # 执行三层筛选获取中间结果
        three_layer_result = self._three_layer_filtering(outer_fold_results, model_name)
        
        # 检查是否为LOCO模式
        is_loco = self._is_loco_mode(outer_fold_results)
        
        if is_loco:
            # LOCO模式：计算队列筛选统计信息
            cohort_stats = self._get_cohort_cv_stats(outer_fold_results, model_name)
            
            return {
                'inner_cv_filtering': {
                    'features_count': len(three_layer_result['inner_cv_features']),
                    'features': three_layer_result['inner_cv_features']
                },
                'outer_cv_filtering': {
                    'features_count': len(three_layer_result['outer_cv_features']),
                    'features': three_layer_result['outer_cv_features']
                },
                'repeat_filtering': {
                    'features_count': len(three_layer_result['repeat_features']),
                    'features': three_layer_result['repeat_features']
                },
                'cohort_cv_stats': cohort_stats
            }
        else:
            # 重复CV模式：计算重复筛选统计信息
            repeat_stats = self._get_repeat_cv_stats(outer_fold_results, model_name)
            
            return {
                'inner_cv_filtering': {
                    'features_count': len(three_layer_result['inner_cv_features']),
                    'features': three_layer_result['inner_cv_features']
                },
                'outer_cv_filtering': {
                    'features_count': len(three_layer_result['outer_cv_features']),
                    'features': three_layer_result['outer_cv_features']
                },
                'repeat_filtering': {
                    'features_count': len(three_layer_result['repeat_features']),
                    'features': three_layer_result['repeat_features']
                },
                'repeat_cv_stats': repeat_stats
            }
    
    def _get_repeat_cv_stats(self, outer_fold_results: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        """获取重复CV特有的统计信息"""
        # 按重复分组
        repeat_groups = {}
        for fold_result in outer_fold_results:
            repeat_idx = fold_result.get('repeat_idx', 0)
            if repeat_idx not in repeat_groups:
                repeat_groups[repeat_idx] = []
            
            fold_features = fold_result['model_results'][model_name]['selected_features']
            repeat_groups[repeat_idx].extend(fold_features)
        
        # 统计每个特征在重复中的稳定性
        feature_repeat_stability = {}
        for repeat_idx, features in repeat_groups.items():
            unique_features = set(features)
            for feature in unique_features:
                if feature not in feature_repeat_stability:
                    feature_repeat_stability[feature] = 0
                feature_repeat_stability[feature] += 1
        
        # 计算重复稳定性分布
        repeat_count = len(repeat_groups)
        stability_distribution = {
            'high_stability': 0,  # 在所有重复中都出现
            'medium_stability': 0,  # 在大部分重复中出现
            'low_stability': 0   # 在少数重复中出现
        }
        
        for feature, count in feature_repeat_stability.items():
            if count == repeat_count:
                stability_distribution['high_stability'] += 1
            elif count >= repeat_count * 0.5:
                stability_distribution['medium_stability'] += 1
            else:
                stability_distribution['low_stability'] += 1
        
        return {
            'n_repeats': repeat_count,
            'feature_repeat_stability': feature_repeat_stability,
            'stability_distribution': stability_distribution,
            'repeat_groups_summary': {
                f'repeat_{i}': len(set(features)) for i, features in repeat_groups.items()
            }
        }
    
    def _get_cohort_cv_stats(self, outer_fold_results: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        """获取LOCO队列CV特有的统计信息"""
        # 统计每个特征在队列中的稳定性
        feature_cohort_stability = {}
        for fold_result in outer_fold_results:
            fold_features = fold_result['model_results'][model_name]['selected_features']
            unique_features = set(fold_features)
            for feature in unique_features:
                if feature not in feature_cohort_stability:
                    feature_cohort_stability[feature] = 0
                feature_cohort_stability[feature] += 1
        
        # 计算队列稳定性分布
        cohort_count = len(outer_fold_results)
        stability_distribution = {
            'high_stability': 0,  # 在所有队列中都出现
            'medium_stability': 0,  # 在大部分队列中出现
            'low_stability': 0   # 在少数队列中出现
        }
        
        for feature, count in feature_cohort_stability.items():
            if count == cohort_count:
                stability_distribution['high_stability'] += 1
            elif count >= cohort_count * 0.5:
                stability_distribution['medium_stability'] += 1
            else:
                stability_distribution['low_stability'] += 1
        
        return {
            'n_cohorts': cohort_count,
            'feature_cohort_stability': feature_cohort_stability,
            'stability_distribution': stability_distribution,
            'cohort_groups_summary': {
                f'cohort_{i}': len(set(fold_result['model_results'][model_name]['selected_features'])) 
                for i, fold_result in enumerate(outer_fold_results)
            }
        }
    
    def _weighted_voting(self, outer_fold_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """加权投票策略。"""
        self.logger.info("使用加权投票策略选择共识特征")
        
        consensus_features = {}
        
        for model_name in outer_fold_results[0]['model_results'].keys():
            # 收集特征和重要性
            feature_importance = {}
            feature_counts = Counter()
            
            for fold_result in outer_fold_results:
                features = fold_result['model_results'][model_name]['selected_features']
                importances = fold_result['model_results'][model_name].get('feature_importance', {})
                
                for feature in features:
                    feature_counts[feature] += 1
                    if feature in importances:
                        if feature not in feature_importance:
                            feature_importance[feature] = []
                        feature_importance[feature].append(importances[feature])
            
            # 计算加权分数
            feature_scores = {}
            n_folds = len(outer_fold_results)
            
            for feature, count in feature_counts.items():
                # 基础分数：出现频率
                frequency_score = count / n_folds
                
                # 重要性分数：平均重要性
                if feature in feature_importance:
                    importance_score = np.mean(feature_importance[feature])
                else:
                    importance_score = 0.0
                
                # 加权分数
                feature_scores[feature] = 0.7 * frequency_score + 0.3 * importance_score
            
            # 选择共识特征
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            consensus_features[model_name] = [
                feature for feature, score in sorted_features
                if score >= self.config.threshold
            ]
            
            # 记录统计信息
            self.selection_stats_[model_name] = {
                'total_features': len(feature_scores),
                'consensus_features': len(consensus_features[model_name]),
                'threshold': self.config.threshold,
                'feature_scores': feature_scores
            }
            
            self.logger.info(f"{model_name}: {len(consensus_features[model_name])} 个共识特征")
        
        return consensus_features
    
    def _stability_score(self, outer_fold_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """稳定性评分策略。"""
        self.logger.info("使用稳定性评分策略选择共识特征")
        
        consensus_features = {}
        
        for model_name in outer_fold_results[0]['model_results'].keys():
            # 收集所有特征
            all_features = set()
            for fold_result in outer_fold_results:
                features = fold_result['model_results'][model_name]['selected_features']
                all_features.update(features)
            
            # 计算稳定性分数
            feature_stability = {}
            n_folds = len(outer_fold_results)
            
            for feature in all_features:
                # 计算特征在多少折中被选中
                selected_count = sum(
                    1 for fold_result in outer_fold_results
                    if feature in fold_result['model_results'][model_name]['selected_features']
                )
                
                # 稳定性分数：选中次数 / 总折数
                stability_score = selected_count / n_folds
                feature_stability[feature] = stability_score
            
            # 选择稳定性高的特征
            consensus_features[model_name] = [
                feature for feature, stability in feature_stability.items()
                if stability >= self.config.stability_threshold
            ]
            
            # 记录统计信息
            self.selection_stats_[model_name] = {
                'total_features': len(feature_stability),
                'consensus_features': len(consensus_features[model_name]),
                'stability_threshold': self.config.stability_threshold,
                'feature_stability': feature_stability
            }
            
            self.logger.info(f"{model_name}: {len(consensus_features[model_name])} 个共识特征")
        
        return consensus_features
    
    def _importance_based(self, outer_fold_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """基于重要性的策略。"""
        self.logger.info("使用基于重要性的策略选择共识特征")
        
        consensus_features = {}
        
        for model_name in outer_fold_results[0]['model_results'].keys():
            # 收集特征重要性
            feature_importance = {}
            feature_counts = Counter()
            
            for fold_result in outer_fold_results:
                features = fold_result['model_results'][model_name]['selected_features']
                importances = fold_result['model_results'][model_name].get('feature_importance', {})
                
                for feature in features:
                    feature_counts[feature] += 1
                    if feature in importances:
                        if feature not in feature_importance:
                            feature_importance[feature] = []
                        feature_importance[feature].append(importances[feature])
            
            # 计算平均重要性
            avg_importance = {}
            for feature, importances in feature_importance.items():
                avg_importance[feature] = np.mean(importances)
            
            # 按重要性排序
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            # 选择重要性高的特征
            n_features = min(
                len(sorted_features),
                max(self.config.min_features, int(len(sorted_features) * self.config.adaptive_ratio))
            )
            
            consensus_features[model_name] = [
                feature for feature, importance in sorted_features[:n_features]
                if importance >= self.config.importance_threshold
            ]
            
            # 记录统计信息
            self.selection_stats_[model_name] = {
                'total_features': len(avg_importance),
                'consensus_features': len(consensus_features[model_name]),
                'importance_threshold': self.config.importance_threshold,
                'avg_importance': avg_importance
            }
            
            self.logger.info(f"{model_name}: {len(consensus_features[model_name])} 个共识特征")
        
        return consensus_features
    
    def _adaptive_selection(self, outer_fold_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """自适应策略。"""
        self.logger.info("使用自适应策略选择共识特征")
        
        consensus_features = {}
        
        for model_name in outer_fold_results[0]['model_results'].keys():
            # 收集特征信息
            all_features = set()
            feature_importance = {}
            feature_counts = Counter()
            
            for fold_result in outer_fold_results:
                features = fold_result['model_results'][model_name]['selected_features']
                importances = fold_result['model_results'][model_name].get('feature_importance', {})
                
                all_features.update(features)
                for feature in features:
                    feature_counts[feature] += 1
                    if feature in importances:
                        if feature not in feature_importance:
                            feature_importance[feature] = []
                        feature_importance[feature].append(importances[feature])
            
            # 计算综合分数
            feature_scores = {}
            n_folds = len(outer_fold_results)
            
            for feature in all_features:
                # 频率分数
                frequency_score = feature_counts[feature] / n_folds
                
                # 重要性分数
                if feature in feature_importance:
                    importance_score = np.mean(feature_importance[feature])
                else:
                    importance_score = 0.0
                
                # 稳定性分数
                stability_score = frequency_score
                
                # 综合分数（自适应权重）
                if len(all_features) > 100:  # 高维数据，更重视稳定性
                    feature_scores[feature] = 0.5 * frequency_score + 0.3 * stability_score + 0.2 * importance_score
                else:  # 低维数据，更重视重要性
                    feature_scores[feature] = 0.3 * frequency_score + 0.2 * stability_score + 0.5 * importance_score
            
            # 自适应阈值
            scores = list(feature_scores.values())
            if len(scores) > 0:
                # 使用分位数作为阈值
                threshold = np.percentile(scores, 70)  # 选择前30%的特征
            else:
                threshold = 0.5
            
            # 选择共识特征
            consensus_features[model_name] = [
                feature for feature, score in feature_scores.items()
                if score >= threshold
            ]
            
            # 确保特征数量在合理范围内
            if len(consensus_features[model_name]) < self.config.min_features:
                # 如果特征太少，降低阈值
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                consensus_features[model_name] = [
                    feature for feature, score in sorted_features[:self.config.min_features]
                ]
            elif len(consensus_features[model_name]) > self.config.max_features:
                # 如果特征太多，提高阈值
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                consensus_features[model_name] = [
                    feature for feature, score in sorted_features[:self.config.max_features]
                ]
            
            # 记录统计信息
            self.selection_stats_[model_name] = {
                'total_features': len(feature_scores),
                'consensus_features': len(consensus_features[model_name]),
                'adaptive_threshold': threshold,
                'feature_scores': feature_scores
            }
            
            self.logger.info(f"{model_name}: {len(consensus_features[model_name])} 个共识特征")
        
        return consensus_features
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """获取选择统计信息。"""
        return self.selection_stats_
    
    def get_feature_scores(self) -> Dict[str, Dict[str, float]]:
        """获取特征分数。"""
        return self.feature_scores_


def create_consensus_selector(
    strategy: str = "majority_voting",
    threshold: float = 0.5,
    min_features: int = 10,
    max_features: int = 1000,
    **kwargs
) -> ConsensusFeatureSelector:
    """
    创建共识特征选择器。
    
    Args:
        strategy: 选择策略
        threshold: 阈值
        min_features: 最小特征数
        max_features: 最大特征数
        **kwargs: 其他配置参数
        
    Returns:
        共识特征选择器实例
    """
    config = ConsensusConfig(
        strategy=ConsensusStrategy(strategy),
        threshold=threshold,
        min_features=min_features,
        max_features=max_features,
        **kwargs
    )
    
    return ConsensusFeatureSelector(config)
