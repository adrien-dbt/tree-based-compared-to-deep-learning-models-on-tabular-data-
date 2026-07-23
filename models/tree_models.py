"""
Tree-based models for tabular data benchmark.
Includes Random Forest, XGBoost, LightGBM, and CatBoost.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class TreeBasedModel:
    """Base class for tree-based models."""
    
    def __init__(self, model_type: str = 'random_forest', task_type: str = 'classification'):
        self.model_type = model_type
        self.task_type = task_type
        self.model = None
        self.is_classifier = task_type == 'classification'
    
    def _get_model(self, **kwargs):
        """Create model instance with default parameters."""
        if self.model_type == 'random_forest':
            if self.is_classifier:
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                    **kwargs
                )
            else:
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                    **kwargs
                )
        
        elif self.model_type == 'xgboost':
            if self.is_classifier:
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    **kwargs
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    **kwargs
                )
        
        elif self.model_type == 'lightgbm':
            if self.is_classifier:
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=-1,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    **kwargs
                )
            else:
                return lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=-1,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    **kwargs
                )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **fit_params):
        """Train the model."""
        self.model = self._get_model()
        
        # Handle categorical features for tree models
        X_train_processed = X_train.copy()
        self.cat_features = X_train_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.cat_features:
            X_train_processed[col] = X_train_processed[col].astype('category').cat.codes
        
        self.model.fit(X_train_processed, y_train, **fit_params)
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_test_processed = X_test.copy()
        
        for col in self.cat_features:
            if col in X_test_processed.columns:
                X_test_processed[col] = X_test_processed[col].astype('category').cat.codes
        
        return self.model.predict(X_test_processed)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classification")
        
        X_test_processed = X_test.copy()
        
        for col in self.cat_features:
            if col in X_test_processed.columns:
                X_test_processed[col] = X_test_processed[col].astype('category').cat.codes
        
        return self.model.predict_proba(X_test_processed)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        
        metrics = {}
        
        if self.is_classifier:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            
            # For binary classification, compute ROC AUC
            if len(np.unique(y_test)) == 2:
                try:
                    proba = self.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, proba)
                except:
                    pass
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
            metrics['r2'] = r2_score(y_test, predictions)
        
        return metrics


class RandomForest(TreeBasedModel):
    """Random Forest model."""
    def __init__(self, task_type: str = 'classification'):
        super().__init__('random_forest', task_type)


class XGBoost(TreeBasedModel):
    """XGBoost model."""
    def __init__(self, task_type: str = 'classification'):
        super().__init__('xgboost', task_type)


class LightGBM(TreeBasedModel):
    """LightGBM model."""
    def __init__(self, task_type: str = 'classification'):
        super().__init__('lightgbm', task_type)


# Default parameters for quick experiments (no hyperparameter tuning)
DEFAULT_PARAMS = {
    'random_forest': {
        'classification': {'n_estimators': 100, 'max_depth': None},
        'regression': {'n_estimators': 100, 'max_depth': None}
    },
    'xgboost': {
        'classification': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
        'regression': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    },
    'lightgbm': {
        'classification': {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.1},
        'regression': {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.1}
    }
}


if __name__ == "__main__":
    # Test tree models
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    for model_name in ['random_forest', 'xgboost', 'lightgbm']:
        print(f"\nTesting {model_name}...")
        model = TreeBasedModel(model_name, 'classification')
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        print(f"Metrics: {metrics}")
