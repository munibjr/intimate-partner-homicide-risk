"""Load synthetic case histories from JSON or CSV."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


class CaseLoader:
    """Load raw case histories for CLI and scripts."""

    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load cases from a JSON or CSV file."""
        file_path = Path(path)
        if file_path.suffix.lower() == ".json":
            return self.load_json(file_path)
        if file_path.suffix.lower() == ".csv":
            return self.load_csv(file_path)
        raise ValueError(f"Unsupported case file type: {file_path.suffix}")

    def load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load a single case or list of cases from JSON."""
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        raise ValueError("JSON case file must contain an object or a list of objects")

    def load_csv(self, path: Path) -> List[Dict[str, Any]]:
        """Load cases from CSV rows with an events JSON column."""
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        cases: List[Dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            events_value = row.get("events") or row.get("events_json")
            if not events_value:
                raise ValueError("CSV input requires an events or events_json column")
            case = {
                "case_id": row.get("case_id") or f"CSV_CASE_{index:03d}",
                "start_date": row.get("start_date"),
                "events": json.loads(events_value),
                "outcome": row.get("outcome", "unknown"),
                "risk_category": row.get("risk_category", "low"),
            }
            cases.append(case)
        return cases

