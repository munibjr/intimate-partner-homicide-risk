"""Evaluation metrics for risk assessment model."""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
from typing import Dict, Tuple


class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        
        metrics = {
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_proba[:, 1])),
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Rates
        metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics['fnr'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    @staticmethod
    def print_results(metrics: Dict[str, float]):
        """Print evaluation results."""
        print("\n=== Model Performance ===")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"\nFalse Positive Rate: {metrics['fpr']:.3f}")
        print(f"False Negative Rate: {metrics['fnr']:.3f}")
