"""Estimate how risk evolves over a case timeline."""

from typing import Any, Dict, List

from src.data.feature_extractor import FeatureExtractor


class RiskTrajectory:
    """Build incremental feature-based trajectory points."""

    def __init__(self) -> None:
        """Create a trajectory analyzer."""
        self.extractor = FeatureExtractor()

    def build(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return a simple cumulative trajectory over events."""
        events = sorted(case.get("events", []), key=lambda event: event.get("date", ""))
        trajectory: List[Dict[str, Any]] = []
        for index in range(1, len(events) + 1):
            partial_case = {**case, "events": events[:index]}
            features = self.extractor.extract_features(partial_case)
            heuristic_score = self._heuristic_score(features)
            trajectory.append(
                {
                    "date": events[index - 1].get("date"),
                    "event_type": events[index - 1].get("type"),
                    "trajectory_score": heuristic_score,
                }
            )
        return trajectory

    def _heuristic_score(self, features: Dict[str, float]) -> float:
        """Calculate a lightweight trajectory score without training a model."""
        weights = {
            "violence_escalation_rate": 20,
            "threat_severity_max": 20,
            "weapon_mention": 20,
            "restraining_order_breaches": 15,
            "isolation_indicators": 10,
            "multiple_agencies_involved": 10,
            "trend_worsening": 5,
        }
        score = sum(features[name] * weight for name, weight in weights.items())
        return round(min(100.0, score), 2)

