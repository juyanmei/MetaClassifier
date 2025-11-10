"""
Helper utilities for metaClassifier.

This module contains various helper functions and utilities.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_object(obj: Any, path: Union[str, Path]) -> None:
    """Save object to file using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: Union[str, Path]) -> Any:
    """Load object from file using joblib."""
    return joblib.load(path)


def create_summary_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for a DataFrame."""
    return {
        'shape': data.shape,
        'dtypes': data.dtypes.value_counts().to_dict(),
        'missing_values': data.isnull().sum().sum(),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object', 'category']).columns.tolist(),
        'memory_usage': data.memory_usage(deep=True).sum(),
    }


def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """Check data quality and return issues."""
    issues = []
    
    # Check for missing values
    missing_cols = data.columns[data.isnull().any()].tolist()
    if missing_cols:
        issues.append(f"Missing values in columns: {missing_cols}")
    
    # Check for infinite values
    inf_cols = data.columns[np.isinf(data.select_dtypes(include=[np.number])).any()].tolist()
    if inf_cols:
        issues.append(f"Infinite values in columns: {inf_cols}")
    
    # Check for constant columns
    constant_cols = data.columns[data.nunique() <= 1].tolist()
    if constant_cols:
        issues.append(f"Constant columns: {constant_cols}")
    
    return {
        'has_issues': len(issues) > 0,
        'issues': issues,
        'missing_columns': missing_cols,
        'infinite_columns': inf_cols,
        'constant_columns': constant_cols,
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator
