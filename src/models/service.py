"""Shared model runtime for CLI and API predictions."""

from typing import Any, Dict, List, Optional

import numpy as np

from src.data.case_generator import CaseGenerator
from src.data.feature_extractor import FEATURE_NAMES, FeatureExtractor
from src.models.ensemble import RiskEnsemble


DISCLAIMER = (
    "Synthetic demo only. This score is decision support for case prioritization "
    "and must not be used as an automatic action trigger or prediction about a person."
)


class RiskPredictionService:
    """Train and serve the synthetic risk model in memory."""

    def __init__(self, n_cases: int = 500, high_risk_ratio: float = 0.3, seed: int = 42):
        """Configure a model service that trains lazily on synthetic data."""
        self.n_cases = n_cases
        self.high_risk_ratio = high_risk_ratio
        self.seed = seed
        self.extractor = FeatureExtractor()
        self.model: Optional[RiskEnsemble] = None
        self.training_metrics: Dict[str, Any] = {}

    def train(self) -> Dict[str, Any]:
        """Train the ensemble on synthetic data and return CV metrics."""
        generator = CaseGenerator(seed=self.seed)
        cases = generator.generate_dataset(
            n_cases=self.n_cases,
            high_risk_ratio=self.high_risk_ratio,
        )
        feature_rows = [self.extractor.extract_features(case) for case in cases]
        x_train = np.array([[row[name] for name in FEATURE_NAMES] for row in feature_rows])
        y_train = np.array([row["risk_label"] for row in feature_rows])

        self.model = RiskEnsemble(random_state=self.seed)
        self.training_metrics = self.model.fit(x_train, y_train)
        return self.training_metrics

    def ensure_trained(self) -> None:
        """Train the model if it has not been trained yet."""
        if self.model is None:
            self.train()

    def predict_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Predict a risk report for a single raw case."""
        self.ensure_trained()
        assert self.model is not None

        features = self.extractor.extract_features(case)
        x_case = np.array([[features[name] for name in FEATURE_NAMES]])
        risk_score = float(self.model.get_risk_scores(x_case)[0])

        return {
            "case_id": case.get("case_id", "unknown"),
            "risk_score": round(risk_score, 2),
            "risk_level": self._risk_level(risk_score),
            "features": {name: features[name] for name in FEATURE_NAMES},
            "top_factors": self._top_factors(features),
            "recommended_actions": self._recommended_actions(risk_score),
            "disclaimer": DISCLAIMER,
        }

    def predict_batch(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict risk reports for multiple raw cases."""
        return [self.predict_case(case) for case in cases]

    def health(self) -> Dict[str, Any]:
        """Return runtime health metadata."""
        return {
            "status": "ok",
            "model_trained": self.model is not None,
            "feature_count": len(FEATURE_NAMES),
            "disclaimer": DISCLAIMER,
        }

    def _top_factors(self, features: Dict[str, float], limit: int = 5) -> List[Dict[str, Any]]:
        """Return the highest-valued interpretable factors."""
        ranked = sorted(FEATURE_NAMES, key=lambda name: features[name], reverse=True)
        return [
            {
                "feature": name,
                "value": round(float(features[name]), 4),
                "direction": "increases review priority",
            }
            for name in ranked[:limit]
            if features[name] > 0
        ]

    def _risk_level(self, risk_score: float) -> str:
        """Map a score to a human-readable review band."""
        if risk_score >= 70:
            return "HIGH"
        if risk_score >= 40:
            return "MEDIUM"
        return "LOW"

    def _recommended_actions(self, risk_score: float) -> List[str]:
        """Return non-automated decision-support recommendations."""
        if risk_score >= 70:
            return [
                "Immediate human review by a specialist team",
                "Victim safety planning check",
                "Multi-agency coordination review",
            ]
        if risk_score >= 40:
            return [
                "Human review of protective order status",
                "Safety check-in and support service review",
            ]
        return [
            "Standard monitoring procedures",
            "Confirm support resources are known",
        ]

