"""
Main MetaClassifier class that orchestrates the entire classification pipeline.

This is the main entry point for the metaClassifier framework.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from .base import BaseClassifier, ExperimentConfig, TaskType, CVStrategy
from .cv_evaluator import CVEvaluatorFactory
from .feature_selector import FeatureSelectorFactory
from .hyperparameter_tuner import HyperparameterTunerFactory
from ..data.loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from ..data.validator import DataValidator
from ..models import ModelFactory
from ..preprocessing.variance_filter import AdaptiveVarianceFilter
from ..evaluation.metrics import MetricsCalculator
from ..evaluation.reporter import ResultsReporter
from ..utils.logger import get_logger


class MetaClassifier(BaseClassifier):
    """
    Main MetaClassifier class that orchestrates the entire classification pipeline.
    
    This class provides a high-level interface for microbiome classification with
    integrated cross-validation, feature selection, and hyperparameter tuning.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the MetaClassifier.
        
        Args:
            config: Complete experiment configuration
        """
        super().__init__(config)
        self.logger = get_logger("MetaClassifier")
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()
        self.data_validator = DataValidator()
        self.adaptive_filter = AdaptiveVarianceFilter(config.adaptive_filter)
        self.metrics_calculator = MetricsCalculator()
        self.results_reporter = ResultsReporter()
        
        # Initialize factories
        self.cv_factory = CVEvaluatorFactory()
        self.feature_selector_factory = FeatureSelectorFactory()
        self.hyperparameter_tuner_factory = HyperparameterTunerFactory()
        self.model_factory = ModelFactory()
        
        # Initialize evaluator
        self.evaluator_ = self.cv_factory.create_evaluator(config.cv)
        
        self.logger.info(f"MetaClassifier initialized with {config.model.name} model")
        
    def fit(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        cohort_info: Optional[np.ndarray] = None
    ) -> 'MetaClassifier':
        """
        Fit the classifier to the training data.
        
        Args:
            X: Feature matrix
            y: Target labels
            cohort_info: Cohort information for LOCO CV
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Starting model fitting...")
        
        # Validate data
        self.data_validator.validate(X, y, cohort_info)
        
        # Preprocess data
        X_processed = self.data_preprocessor.fit_transform(X, y)
        
        # Apply adaptive variance filtering
        if self.config.adaptive_filter.enabled:
            X_processed, filter_info = self.adaptive_filter.filter_features(X_processed, y)
            self.logger.info(f"Applied adaptive variance filtering: {filter_info}")
        
        # Create model
        model = self.model_factory.create_model(self.config.model)
        
        # Fit model
        model.fit(X_processed, y)
        self.models_[self.config.model.name] = model
        
        self.logger.info("Model fitting completed")
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.models_:
            raise ValueError("Model must be fitted before making predictions")
            
        # Preprocess data
        X_processed = self.data_preprocessor.transform(X)
        
        # Apply adaptive variance filtering
        if self.config.adaptive_filter.enabled:
            X_processed, _ = self.adaptive_filter.filter_features(X_processed)
        
        # Make predictions
        model = list(self.models_.values())[0]  # Use first model
        return model.predict(X_processed)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class probabilities
        """
        if not self.models_:
            raise ValueError("Model must be fitted before making predictions")
            
        # Preprocess data
        X_processed = self.data_preprocessor.transform(X)
        
        # Apply adaptive variance filtering
        if self.config.adaptive_filter.enabled:
            X_processed, _ = self.adaptive_filter.filter_features(X_processed)
        
        # Make predictions
        model = list(self.models_.values())[0]  # Use first model
        return model.predict_proba(X_processed)
        
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        cohort_info: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the classifier using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            cohort_info: Cohort information for LOCO CV
            
        Returns:
            Evaluation results
        """
        self.logger.info("Starting cross-validation evaluation...")
        
        # Validate data
        self.data_validator.validate(X, y, cohort_info)
        
        # Preprocess data
        X_processed = self.data_preprocessor.fit_transform(X, y)
        
        # Create models
        models = {}
        for model_config in [self.config.model]:
            model = self.model_factory.create_model(model_config)
            models[model_config.name] = model
        
        # Run cross-validation
        self.results_ = self.evaluator_.evaluate(
            X_processed, y, models, cohort_info
        )
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(self.results_)
        self.results_['metrics'] = metrics
        
        # Generate report
        if self.config.output_dir:
            report_path = Path(self.config.output_dir) / "evaluation_report.html"
            self.results_reporter.generate_report(self.results_, report_path)
        
        self.logger.info("Cross-validation evaluation completed")
        return self.results_
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance scores or None if not available
        """
        if not self.models_:
            return None
            
        model = list(self.models_.values())[0]
        return model.get_feature_importance()
        
    def get_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest evaluation results.
        
        Returns:
            Evaluation results or None if not available
        """
        return self.results_
        
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        import joblib
        
        model_data = {
            'models': self.models_,
            'preprocessor': self.data_preprocessor,
            'adaptive_filter': self.adaptive_filter,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: Union[str, Path]) -> 'MetaClassifier':
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        import joblib
        
        model_data = joblib.load(path)
        
        self.models_ = model_data['models']
        self.data_preprocessor = model_data['preprocessor']
        self.adaptive_filter = model_data['adaptive_filter']
        self.config = model_data['config']
        
        self.logger.info(f"Model loaded from {path}")
        return self


def create_classifier(
    model_name: str,
    task_type: TaskType = TaskType.CLASSIFICATION,
    cv_strategy: CVStrategy = CVStrategy.REPEATED_KFOLD,
    outer_folds: int = 5,
    inner_folds: int = 3,
    enable_adaptive_filtering: bool = True,
    **kwargs
) -> MetaClassifier:
    """
    Factory function to create a MetaClassifier with common configurations.
    
    Args:
        model_name: Name of the model to use
        task_type: Type of task (classification or regression)
        cv_strategy: Cross-validation strategy
        outer_folds: Number of outer CV folds
        inner_folds: Number of inner CV folds
        enable_adaptive_filtering: Whether to enable adaptive variance filtering
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MetaClassifier instance
    """
    from .base import ModelConfig, CVConfig, AdaptiveFilterConfig, ExperimentConfig
    
    # Create configurations
    model_config = ModelConfig(
        name=model_name,
        task_type=task_type,
        hyperparameters=kwargs.get('model_params', {}),
        feature_selection=kwargs.get('feature_selection', {}),
        preprocessing=kwargs.get('preprocessing', {})
    )
    
    cv_config = CVConfig(
        strategy=cv_strategy,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        random_state=kwargs.get('random_state', 42),
        n_jobs=kwargs.get('n_jobs', 1)
    )
    
    adaptive_filter_config = AdaptiveFilterConfig(
        enabled=enable_adaptive_filtering,
        min_q=kwargs.get('min_q', 0.05),
        max_q=kwargs.get('max_q', 0.2),
        r_mid=kwargs.get('r_mid', 5.0),
        steepness=kwargs.get('steepness', 0.5)
    )
    
    experiment_config = ExperimentConfig(
        model=model_config,
        cv=cv_config,
        adaptive_filter=adaptive_filter_config,
        data_paths=kwargs.get('data_paths', {}),
        output_dir=kwargs.get('output_dir', './results'),
        verbose=kwargs.get('verbose', True)
    )
    
    return MetaClassifier(experiment_config)
