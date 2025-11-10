"""
Data validation utilities for metaClassifier.

This module handles data validation and quality checks.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from ..utils.logger import get_logger


class DataValidator:
    """Data validator for microbiome datasets."""
    
    def __init__(self):
        self.logger = get_logger("DataValidator")
        
    def validate(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        cohort_info: Optional[np.ndarray] = None
    ) -> None:
        """
        Validate input data.
        
        Args:
            X: Feature matrix
            y: Target labels
            cohort_info: Cohort information (optional)
            
        Raises:
            ValueError: If data validation fails
        """
        self.logger.info("Validating input data...")
        
        # Validate feature matrix
        self._validate_feature_matrix(X)
        
        # Validate labels
        self._validate_labels(y)
        
        # Validate cohort info if provided
        if cohort_info is not None:
            self._validate_cohort_info(cohort_info, len(y))
        
        # Validate data consistency
        self._validate_data_consistency(X, y, cohort_info)
        
        self.logger.info("Data validation passed")
        
    def _validate_feature_matrix(self, X: pd.DataFrame) -> None:
        """Validate feature matrix."""
        if X.empty:
            raise ValueError("Feature matrix is empty")
        
        if X.shape[0] == 0:
            raise ValueError("No samples in feature matrix")
        
        if X.shape[1] == 0:
            raise ValueError("No features in feature matrix")
        
        # Check for infinite values
        if np.isinf(X.values).any():
            raise ValueError("Feature matrix contains infinite values")
        
        # Check for NaN values
        if X.isnull().any().any():
            self.logger.warning("Feature matrix contains NaN values")
        
    def _validate_labels(self, y: np.ndarray) -> None:
        """Validate target labels."""
        if len(y) == 0:
            raise ValueError("No labels provided")
        
        if not isinstance(y, np.ndarray):
            raise ValueError("Labels must be a numpy array")
        
        # Check for binary classification
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(f"Expected 2 classes, found {len(unique_labels)}: {unique_labels}")
        
        # Check label values
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("Labels must be 0 and 1 for binary classification")
        
        # Check for class balance
        class_counts = np.bincount(y)
        min_class_count = np.min(class_counts)
        if min_class_count < 2:
            raise ValueError("Each class must have at least 2 samples")
        
    def _validate_cohort_info(self, cohort_info: np.ndarray, expected_length: int) -> None:
        """Validate cohort information."""
        if len(cohort_info) != expected_length:
            raise ValueError(f"Cohort info length ({len(cohort_info)}) doesn't match labels length ({expected_length})")
        
        unique_cohorts = np.unique(cohort_info)
        if len(unique_cohorts) < 2:
            raise ValueError("At least 2 cohorts required for LOCO CV")
        
    def _validate_data_consistency(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        cohort_info: Optional[np.ndarray]
    ) -> None:
        """Validate data consistency."""
        if len(X) != len(y):
            raise ValueError(f"Feature matrix length ({len(X)}) doesn't match labels length ({len(y)})")
        
        if cohort_info is not None and len(cohort_info) != len(y):
            raise ValueError(f"Cohort info length ({len(cohort_info)}) doesn't match labels length ({len(y)})")
