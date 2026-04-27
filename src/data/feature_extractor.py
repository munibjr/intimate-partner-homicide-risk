"""Extract interpretable features from case histories."""

from datetime import datetime
from typing import Any, Dict, List, Optional


FEATURE_NAMES = [
    "violence_incidents",
    "max_violence_severity",
    "violence_escalation_rate",
    "threat_frequency",
    "weapon_mention",
    "threat_severity_max",
    "isolation_indicators",
    "restraining_order_breaches",
    "police_contact_frequency",
    "child_welfare_contact",
    "days_since_last_incident",
    "trend_worsening",
    "velocity_of_escalation",
    "multiple_agencies_involved",
    "breach_of_protection",
]


class FeatureExtractor:
    """Extract domain-expert features from case data."""

    def __init__(self, reference_date: Optional[datetime] = None):
        """Create an extractor with a deterministic reference date."""
        self.reference_date = reference_date or datetime(2023, 1, 1)

    def extract_features(self, case: Dict[str, Any]) -> Dict[str, float]:
        """Extract all features from a case."""
        events = sorted(case["events"], key=lambda event: event.get("date", ""))

        features = {
            # Violence metrics
            "violence_incidents": self._count_violence(events),
            "max_violence_severity": self._max_severity(events),
            "violence_escalation_rate": self._escalation_rate(events),

            # Threat metrics
            "threat_frequency": self._count_threats(events),
            "weapon_mention": float(self._has_weapon_mention(events)),
            "threat_severity_max": self._max_threat_severity(events),

            # Isolation and control
            "isolation_indicators": self._count_isolation(events),

            # System interaction
            "restraining_order_breaches": self._count_breaches(events),
            "police_contact_frequency": self._police_contact_rate(events),
            "child_welfare_contact": float(self._has_barnevern_contact(events)),

            # Temporal dynamics
            "days_since_last_incident": self._days_since_incident(events),
            "trend_worsening": float(self._is_escalating(events)),
            "velocity_of_escalation": self._escalation_velocity(events),

            # Multi-agency
            "multiple_agencies_involved": self._agency_count(events),
            "breach_of_protection": float(self._has_protection_breach(events)),
            "risk_label": 1.0 if case.get("risk_category") == "high" else 0.0,
        }

        return features

    def _count_violence(self, events: List[Dict]) -> float:
        """Count violence incidents normalized to a practical maximum."""
        count = sum(1 for e in events if e.get("type") == "police_report" and e.get("violence"))
        return min(1.0, float(count) / 5.0)

    def _max_severity(self, events: List[Dict]) -> float:
        """Get maximum violence severity (0-10)."""
        violence_events = [e for e in events if e.get("type") == "police_report"]
        if not violence_events:
            return 0.0
        return float(max(e.get("severity", 0) for e in violence_events)) / 10.0

    def _escalation_rate(self, events: List[Dict]) -> float:
        """Calculate if violence is escalating (0-1)."""
        violence_events = [e for e in events if e.get("type") == "police_report" and e.get("violence")]
        if len(violence_events) < 2:
            return 0.0

        severities = [e.get("severity", 0) for e in violence_events]
        if severities[0] == 0:
            return 0.0
        return min(1.0, (severities[-1] - severities[0]) / severities[0])

    def _count_threats(self, events: List[Dict]) -> float:
        """Count threat events normalized to a practical maximum."""
        count = sum(1 for e in events if e.get("type") == "threat")
        return min(1.0, float(count) / 5.0)

    def _has_weapon_mention(self, events: List[Dict]) -> bool:
        """Check if weapon is mentioned."""
        return any(e.get("weapon_mention") for e in events)

    def _max_threat_severity(self, events: List[Dict]) -> float:
        """Get maximum threat severity."""
        threat_events = [e for e in events if e.get("type") == "threat"]
        if not threat_events:
            return 0.0
        return float(max(e.get("severity", 0) for e in threat_events)) / 10.0

    def _count_isolation(self, events: List[Dict]) -> float:
        """Count isolation/control behaviors normalized to a practical maximum."""
        count = sum(1 for e in events if e.get("type") == "isolation_behavior")
        return min(1.0, float(count) / 5.0)

    def _count_breaches(self, events: List[Dict]) -> float:
        """Count restraining order breaches normalized to a practical maximum."""
        count = sum(1 for e in events if e.get("type") == "restraining_order_breach")
        return min(1.0, float(count) / 5.0)

    def _police_contact_rate(self, events: List[Dict]) -> float:
        """Calculate police contact frequency."""
        police_events = sum(1 for e in events if "police" in e.get("type", "").lower())
        return float(police_events) / max(1, len(events))

    def _has_barnevern_contact(self, events: List[Dict]) -> bool:
        """Check if child welfare contacted."""
        return any(e.get("type") == "child_welfare_contact" for e in events)

    def _days_since_incident(self, events: List[Dict]) -> float:
        """Calculate days since last incident."""
        if not events:
            return 0.0
        last_event = events[-1]
        last_date = datetime.fromisoformat(last_event["date"])
        days = max(0, (self.reference_date - last_date).days)
        return min(1.0, days / 365.0)

    def _is_escalating(self, events: List[Dict]) -> bool:
        """Check if case is getting worse."""
        if len(events) < 2:
            return False
        return any(e.get("severity", 0) >= 5 for e in events[-2:])

    def _escalation_velocity(self, events: List[Dict]) -> float:
        """Calculate speed of escalation."""
        if len(events) < 2:
            return 0.0
        first_date = datetime.fromisoformat(events[0]["date"])
        last_date = datetime.fromisoformat(events[-1]["date"])
        days_span = (last_date - first_date).days
        if days_span == 0:
            return 0.0
        return min(1.0, float(len(events)) / days_span * 30.0)

    def _agency_count(self, events: List[Dict]) -> float:
        """Count how many different agencies are involved."""
        agencies = set()
        for e in events:
            event_type = e.get("type", "")
            if "police" in event_type:
                agencies.add("police")
            if "child_welfare" in event_type:
                agencies.add("barnevern")
            if "health" in event_type:
                agencies.add("health")
        return float(len(agencies)) / 3.0  # Normalize to 0-1

    def _has_protection_breach(self, events: List[Dict]) -> bool:
        """Check if a protective order was breached."""
        return any(e.get("type") == "restraining_order_breach" for e in events)
