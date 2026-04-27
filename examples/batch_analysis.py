"""Example: run batch predictions and pattern analysis."""

import json

from src.analysis.pattern_detector import PatternDetector
from src.data.case_generator import CaseGenerator
from src.models.service import RiskPredictionService


def main() -> int:
    """Generate several cases and print predictions with patterns."""
    cases = CaseGenerator(seed=42).generate_dataset(n_cases=6, high_risk_ratio=0.5)
    service = RiskPredictionService(n_cases=120)
    detector = PatternDetector()
    results = []
    for case in cases:
        report = service.predict_case(case)
        report["patterns"] = detector.detect(case)
        results.append(report)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

