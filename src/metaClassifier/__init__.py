"""
metaClassifier v1.0

A comprehensive microbiome classification framework with advanced cross-validation strategies.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.base import BaseClassifier, BaseEvaluator
from .core.cv_evaluator import CVEvaluator, LOCOEvaluator, RepeatedKFoldEvaluator
from .core.nested_cv_evaluator import NestedCVEvaluator
from .core.nested_cv_classifier import NestedCVClassifier, create_nested_cv_classifier
from .core.final_model_trainer import FinalModelTrainer, create_final_model_trainer
from .core.feature_selector import FeatureSelector
from .core.hyperparameter_tuner import HyperparameterTuner

# Data handling
from .data.loader import DataLoader
from .data.preprocessor import DataPreprocessor
from .data.validator import DataValidator

# Models
from .models.base_model import BaseModel
from .models.lasso import LassoClassifier
from .models.random_forest import RandomForestClassifier
from .models.neural_network import NeuralNetworkClassifier

# Preprocessing
from .preprocessing.variance_filter import AdaptiveVarianceFilter
from .preprocessing.clr_transform import CLRTransformer
from .preprocessing.feature_engineering import FeatureEngineer

# Evaluation
from .evaluation.metrics import MetricsCalculator
from .evaluation.visualizer import ResultsVisualizer
from .evaluation.reporter import ResultsReporter

# Main classifier
from .core.meta_classifier import MetaClassifier

__all__ = [
    # Core
    "BaseClassifier",
    "BaseEvaluator", 
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
    
    # Data
    "DataLoader",
    "DataPreprocessor", 
    "DataValidator",
    
    # Models
    "BaseModel",
    "LassoClassifier",
    "RandomForestClassifier",
    "NeuralNetworkClassifier",
    
    # Preprocessing
    "AdaptiveVarianceFilter",
    "CLRTransformer",
    "FeatureEngineer",
    
    # Evaluation
    "MetricsCalculator",
    "ResultsVisualizer",
    "ResultsReporter",
    
    # Main
    "MetaClassifier",
]
