"""Generate user-facing risk reports."""

import json
from typing import Any, Dict, List

from src.models.service import DISCLAIMER


class ReportGenerator:
    """Format prediction outputs for people and machines."""

    def to_json(self, report: Dict[str, Any]) -> str:
        """Serialize a single report as pretty JSON."""
        return json.dumps(report, indent=2, sort_keys=True)

    def batch_to_json(self, reports: List[Dict[str, Any]]) -> str:
        """Serialize batch reports as pretty JSON."""
        return json.dumps({"disclaimer": DISCLAIMER, "predictions": reports}, indent=2, sort_keys=True)

    def to_text(self, report: Dict[str, Any]) -> str:
        """Render a concise human-readable report."""
        factors = "\n".join(
            f"- {factor['feature']}: {factor['value']}"
            for factor in report.get("top_factors", [])
        )
        actions = "\n".join(f"- {action}" for action in report.get("recommended_actions", []))
        return (
            f"Case: {report['case_id']}\n"
            f"Risk score: {report['risk_score']:.2f}\n"
            f"Risk level: {report['risk_level']}\n\n"
            f"Top factors:\n{factors or '- None'}\n\n"
            f"Recommended human-review actions:\n{actions}\n\n"
            f"Disclaimer: {report.get('disclaimer', DISCLAIMER)}"
        )

