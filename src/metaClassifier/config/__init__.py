"""
Configuration modules for metaClassifier.

This module contains default configurations and model-specific configurations.
"""

from .default_config import DEFAULT_CONFIG
from .model_configs import MODEL_CONFIGS
from .feature_selection import FeatureSelectionConfig

__all__ = [
    "DEFAULT_CONFIG",
    "MODEL_CONFIGS",
    "FeatureSelectionConfig",
]
