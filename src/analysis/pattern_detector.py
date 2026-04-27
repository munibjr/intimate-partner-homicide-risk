"""Detect common escalation patterns in synthetic case histories."""

from typing import Any, Dict, List


class PatternDetector:
    """Detect interpretable event patterns."""

    def detect(self, case: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return detected escalation patterns."""
        events = case.get("events", [])
        event_types = [event.get("type", "") for event in events]
        patterns: List[Dict[str, str]] = []

        if "threat" in event_types and any(event.get("weapon_mention") for event in events):
            patterns.append(
                {
                    "pattern": "weapon_related_threat",
                    "description": "Threat history includes a weapon mention.",
                }
            )
        if "restraining_order_breach" in event_types:
            patterns.append(
                {
                    "pattern": "protective_order_breach",
                    "description": "A protective or restraining order breach is present.",
                }
            )
        if event_types.count("police_report") >= 2:
            patterns.append(
                {
                    "pattern": "repeated_police_contact",
                    "description": "Multiple police reports appear in the case history.",
                }
            )
        if "isolation_behavior" in event_types:
            patterns.append(
                {
                    "pattern": "coercive_control_indicator",
                    "description": "The history includes isolation or controlling behavior.",
                }
            )

        return patterns

