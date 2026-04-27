"""Analyze escalation trends in case histories."""

from datetime import datetime
from typing import Any, Dict, List


class TrendAnalyzer:
    """Analyze how severity changes over time."""

    def analyze(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Return trend metrics for a case."""
        events = sorted(case.get("events", []), key=lambda event: event.get("date", ""))
        severity_events = [event for event in events if event.get("severity") is not None]
        if len(severity_events) < 2:
            return {"trend": "insufficient_data", "slope": 0.0, "severity_points": len(severity_events)}

        first = severity_events[0]
        last = severity_events[-1]
        first_date = datetime.fromisoformat(first["date"])
        last_date = datetime.fromisoformat(last["date"])
        days = max(1, (last_date - first_date).days)
        slope = (float(last["severity"]) - float(first["severity"])) / days

        if slope > 0.02:
            trend = "worsening"
        elif slope < -0.02:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(slope, 4),
            "severity_points": len(severity_events),
            "max_severity": max(float(event.get("severity", 0)) for event in severity_events),
        }

