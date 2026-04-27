"""Ensemble model with XGBoost and LightGBM."""

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


class RiskEnsemble:
    """Ensemble model for homicide risk assessment."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        self.xgb = XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            scale_pos_weight=2.3,  # Handle class imbalance
            random_state=random_state,
            verbosity=0,
            eval_metric="logloss",
        )

        if LGBMClassifier is not None:
            self.secondary_model = LGBMClassifier(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=200,
                class_weight='balanced',
                random_state=random_state,
                verbosity=-1
            )
            secondary_name = "lgb"
        else:
            self.secondary_model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                random_state=random_state,
            )
            secondary_name = "sklearn_hgb"

        self.ensemble = VotingClassifier(
            estimators=[
                ('xgb', self.xgb),
                (secondary_name, self.secondary_model),
            ],
            voting='soft',
            weights=[0.6, 0.4]  # XGB slightly higher weight
        )
        
        self.is_trained = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train ensemble model."""
        self.ensemble.fit(X_train, y_train)
        self.is_trained = True
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            self.ensemble, X_train, y_train, cv=cv, scoring='roc_auc'
        )
        
        return {
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist()
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class (0=low-risk, 1=high-risk)."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of high-risk."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.ensemble.predict_proba(X)
    
    def get_risk_scores(self, X: np.ndarray) -> np.ndarray:
        """Get risk scores (0-100)."""
        proba = self.predict_proba(X)
        return proba[:, 1] * 100
