"""Example: run one synthetic prediction."""

import json

from src.data.case_generator import CaseGenerator
from src.models.service import RiskPredictionService


def main() -> int:
    """Generate one case and print a risk report."""
    case = CaseGenerator(seed=42).generate_high_risk_case()
    report = RiskPredictionService(n_cases=120).predict_case(case)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

