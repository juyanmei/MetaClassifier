"""
Model-specific configurations for metaClassifier.

This module contains configurations for different machine learning models.
"""

from typing import Dict, Any, List

MODEL_CONFIGS = {
    "Lasso": {
        "hyperparameters": {
            "cv": 3,
            "random_state": 42,
            "max_iter": 1000,
            "n_jobs": 1
        },
        "feature_selection": {
            "method": "adaptive",
            "max_features": 50
        }
    },
    
    "RandomForest": {
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": 1
        },
        "feature_selection": {
            "method": "selectfrommodel",
            "max_features": 50
        }
    },
    
    "NeuralNetwork": {
        "hyperparameters": {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "random_state": 42
        },
        "feature_selection": {
            "method": "selectkbest",
            "max_features": 50
        }
    },
    
    "SVM": {
        "hyperparameters": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "random_state": 42
        },
        "feature_selection": {
            "method": "selectkbest",
            "max_features": 50
        }
    },
    
    "XGBoost": {
        "hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "random_state": 42,
            "n_jobs": 1
        },
        "feature_selection": {
            "method": "selectfrommodel",
            "max_features": 50
        }
    }
}

# Hyperparameter grids for tuning
HYPERPARAMETER_GRIDS = {
    "Lasso": {
        "cv": [3, 5],
        "max_iter": [1000, 2000]
    },
    
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    
    "NeuralNetwork": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01]
    },
    
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
    
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "subsample": [0.8, 1.0]
    }
}
