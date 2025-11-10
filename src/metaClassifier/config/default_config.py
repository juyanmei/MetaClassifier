"""
Default configuration for metaClassifier.

This module contains the default configuration settings.
"""

from typing import Dict, Any

DEFAULT_CONFIG = {
    # Model configuration
    "model": {
        "name": "Lasso",
        "task_type": "classification",
        "hyperparameters": {
            "cv": 3,
            "random_state": 42,
            "max_iter": 1000,
            "n_jobs": 1
        }
    },
    
    # Cross-validation configuration
    "cv": {
        "strategy": "repeated_kfold",
        "outer_folds": 5,
        "inner_folds": 3,
        "n_repeats": 1,
        "random_state": 42,
        "n_jobs": 1
    },
    
    # Feature selection configuration
    "feature_selection": {
        "method": "adaptive",
        "max_features": 50,
        "strategies": ["selectkbest", "selectfrommodel"]
    },
    
    # Adaptive filtering configuration
    "adaptive_filtering": {
        "enabled": True,
        "min_q": 0.05,
        "max_q": 0.2,
        "r_mid": 5.0,
        "steepness": 0.5
    },
    
    # Data preprocessing configuration
    "preprocessing": {
        "scaling_method": "standard",
        "imputation_strategy": "median",
        "remove_constant_features": True,
        "clr_transform": False
    },
    
    # Evaluation configuration
    "evaluation": {
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "generate_plots": True,
        "save_results": True
    },
    
    # Output configuration
    "output": {
        "directory": "./results",
        "format": "html",
        "verbose": True
    },
    
    # Logging configuration
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
        "file": None
    }
}
