from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from src.utils.wandb_utils import log_metrics

class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def build_model(self) -> Any:
        """Build the model."""
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        if not self.is_fitted:
            self.model = self.build_model()
        
        scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring='neg_mean_squared_error'
        )
        
        cv_results = {
            'cv_rmse_mean': np.sqrt(-scores.mean()),
            'cv_rmse_std': np.sqrt(scores.std()),
        }
        
        # Log to wandb
        log_metrics(cv_results)
        
        return cv_results
    
    def save_model(self, path: str) -> None:
        """Save the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        joblib.dump(self.model, path)
    
    def load_model(self, path: str) -> None:
        """Load the model."""
        import joblib
        self.model = joblib.load(path)
        self.is_fitted = True