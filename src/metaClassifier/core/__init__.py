"""
Core functionality for metaClassifier.

This module contains the fundamental classes and interfaces for the metaClassifier framework.
"""

from .base import BaseClassifier, BaseEvaluator, BaseModel
from .cv_evaluator import CVEvaluator, LOCOEvaluator, RepeatedKFoldEvaluator
from .nested_cv_evaluator import NestedCVEvaluator
from .nested_cv_classifier import NestedCVClassifier, create_nested_cv_classifier
from .final_model_trainer import FinalModelTrainer, create_final_model_trainer
from .feature_selector import FeatureSelector
from .hyperparameter_tuner import HyperparameterTuner
from .consensus_feature_selector import ConsensusFeatureSelector, create_consensus_selector
from .model_specific_feature_selector import ModelSpecificFeatureSelector
from .model_specific_hyperparameter_tuner import ModelSpecificHyperparameterTuner
from .standard_output_manager import StandardOutputManager
from .meta_classifier import MetaClassifier

__all__ = [
    "BaseClassifier",
    "BaseEvaluator", 
    "BaseModel",
    "CVEvaluator",
    "LOCOEvaluator",
    "RepeatedKFoldEvaluator",
    "NestedCVEvaluator",
    "NestedCVClassifier",
    "create_nested_cv_classifier",
    "FinalModelTrainer",
    "create_final_model_trainer",
    "FeatureSelector",
    "HyperparameterTuner",
    "ConsensusFeatureSelector",
    "create_consensus_selector",
    "ModelSpecificFeatureSelector",
    "ModelSpecificHyperparameterTuner",
    "StandardOutputManager",
    "MetaClassifier",
]
