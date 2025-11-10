"""
特征选择配置 for metaClassifier v1.0.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class FeatureSelectionConfig:
    """特征选择配置。"""
    enabled: bool = True
    # 移除未使用的 strategy 与 top_n，保留 threshold/search_method
    threshold: float = 0.5
    search_method: str = 'grid'  # 'grid', 'random', 'bayes'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            'enabled': self.enabled,
            'strategy': self.strategy,
            'top_n': self.top_n,
            'threshold': self.threshold,
            'search_method': self.search_method
        }
