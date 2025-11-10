"""
CLR (Centered Log-Ratio) transformation for metaClassifier.

This module contains the CLR transformation implementation for compositional data.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from ..core.base import BasePreprocessor
from ..utils.logger import get_logger


class CLRTransformer(BasePreprocessor):
    """CLR (Centered Log-Ratio) transformer for compositional data."""
    
    def __init__(self, 
                 pseudocount: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.pseudocount = pseudocount
        self.logger = get_logger("CLRTransformer")
        self.geometric_mean_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'CLRTransformer':
        """Fit the CLR transformer to the data."""
        self.logger.info("Fitting CLR transformer...")
        
        # Add pseudocount to avoid log(0)
        X_pseudo = X + self.pseudocount
        
        # Calculate geometric mean for each sample
        self.geometric_mean_ = np.exp(np.mean(np.log(X_pseudo), axis=1))
        
        self.is_fitted = True
        self.logger.info("CLR transformer fitted successfully")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using CLR transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transforming")
            
        self.logger.info("Applying CLR transformation...")
        
        # Add pseudocount to avoid log(0)
        X_pseudo = X + self.pseudocount
        
        # Apply CLR transformation
        X_clr = np.log(X_pseudo.div(self.geometric_mean_, axis=0))
        
        # Return as DataFrame with original column names
        X_clr_df = pd.DataFrame(X_clr, columns=X.columns, index=X.index)
        
        self.logger.info("CLR transformation completed")
        return X_clr_df
