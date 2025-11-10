"""
Model implementations for metaClassifier.

This module contains various machine learning model implementations.
"""

from .base_model import BaseModel
from .lasso import LassoClassifier
from .random_forest import RandomForestClassifier
from .neural_network import NeuralNetworkClassifier
from .elastic_net import ElasticNetClassifier
from .xgboost import XGBoostClassifier
from .catboost import CatBoostClassifier
from .svm import SVMClassifier
from .logistic_regression import LogisticRegressionClassifier
from .knn import KNNClassifier
from .gaussian_nb import GaussianNBClassifier


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_model(model_config):
        """Create a model based on configuration."""
        model_name = model_config.name.lower()
        
        if model_name == 'lasso':
            return LassoClassifier(**model_config.hyperparameters)
        elif model_name == 'randomforest':
            return RandomForestClassifier(**model_config.hyperparameters)
        elif model_name == 'neuralnetwork':
            return NeuralNetworkClassifier(**model_config.hyperparameters)
        elif model_name == 'elasticnet':
            return ElasticNetClassifier(**model_config.hyperparameters)
        elif model_name == 'xgboost':
            return XGBoostClassifier(**model_config.hyperparameters)
        elif model_name == 'catboost':
            return CatBoostClassifier(**model_config.hyperparameters)
        elif model_name == 'svm':
            return SVMClassifier(**model_config.hyperparameters)
        elif model_name == 'logistic':
            return LogisticRegressionClassifier(**model_config.hyperparameters)
        elif model_name == 'knn':
            return KNNClassifier(**model_config.hyperparameters)
        elif model_name == 'gaussiannb':
            return GaussianNBClassifier(**model_config.hyperparameters)
        else:
            raise ValueError(f"Unknown model: {model_name}")


__all__ = [
    "BaseModel",
    "LassoClassifier",
    "RandomForestClassifier", 
    "NeuralNetworkClassifier",
    "ElasticNetClassifier",
    "XGBoostClassifier",
    "CatBoostClassifier",
    "SVMClassifier",
    "KNNClassifier",
    "GaussianNBClassifier",
    "ModelFactory",
    "LogisticRegressionClassifier",
]
