"""SHAP-based explainability for model predictions."""

import shap
import numpy as np
from typing import Dict, List, Any


class RiskExplainer:
    """Explain model predictions using SHAP."""
    
    def __init__(self, model, X_train: np.ndarray, feature_names: List[str]):
        """Initialize explainer."""
        self.model = model
        self.feature_names = feature_names
        
        # Create SHAP explainer
        print("Initializing SHAP explainer...")
        self.explainer = shap.Explainer(
            self.model.predict_proba,
            X_train,
            feature_names=feature_names
        )
    
    def explain_case(self, X: np.ndarray, case_id: str) -> Dict[str, Any]:
        """Generate explanation for a single case."""
        if X.shape[0] != 1:
            raise ValueError("Explain one case at a time")
        
        # Get prediction
        risk_score = self.model.get_risk_scores(X)[0]
        
        # Get SHAP values
        shap_values = self.explainer(X)
        
        # Extract top factors
        shap_vals = shap_values.values[0]
        shap_base = shap_values.base_values[0]
        
        # Sort by absolute importance
        importance_idx = np.argsort(np.abs(shap_vals))[::-1]
        
        # Top factors increasing risk
        increasing_factors = []
        decreasing_factors = []
        
        for idx in importance_idx[:5]:
            feature_name = self.feature_names[idx]
            shap_value = shap_vals[idx]
            feature_value = X[0, idx]
            
            if shap_value > 0:
                increasing_factors.append({
                    'feature': feature_name,
                    'value': float(feature_value),
                    'contribution': f"+{abs(shap_value)*100:.1f}%",
                    'direction': 'increases risk'
                })
            else:
                decreasing_factors.append({
                    'feature': feature_name,
                    'value': float(feature_value),
                    'contribution': f"-{abs(shap_value)*100:.1f}%",
                    'direction': 'decreases risk'
                })
        
        # Categorize risk
        if risk_score >= 70:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'case_id': case_id,
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'base_value': float(shap_base),
            'top_factors_increasing_risk': increasing_factors,
            'top_factors_decreasing_risk': decreasing_factors,
            'recommended_actions': self._get_recommendations(risk_score)
        }
    
    def _get_recommendations(self, risk_score: float) -> List[str]:
        """Get recommended actions based on risk score."""
        if risk_score >= 70:
            return [
                "Immediate protective order review",
                "Victim safety planning session",
                "Multi-agency coordination meeting",
                "Specialized DV unit consultation",
                "Consider escalation to specialized team"
            ]
        elif risk_score >= 40:
            return [
                "Review protective order status",
                "Victim safety check-in",
                "Increase monitoring frequency",
                "Consider additional support services"
            ]
        else:
            return [
                "Standard monitoring procedures",
                "Ensure victim knows resources available"
            ]
