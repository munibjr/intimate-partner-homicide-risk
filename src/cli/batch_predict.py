"""Run batch risk predictions from JSON or CSV."""

import argparse
from typing import Optional

from src.models.service import RiskPredictionService
from src.utils.case_loader import CaseLoader
from src.utils.report_generator import ReportGenerator


def build_parser() -> argparse.ArgumentParser:
    """Build the batch CLI parser."""
    parser = argparse.ArgumentParser(description="Predict review-priority risk for many synthetic cases.")
    parser.add_argument("case_file", help="Path to a JSON or CSV file containing cases")
    parser.add_argument("--format", choices=["json"], default="json", help="Output format")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the batch prediction CLI."""
    args = build_parser().parse_args(argv)
    cases = CaseLoader().load(args.case_file)
    service = RiskPredictionService()
    reports = service.predict_batch(cases)
    print(ReportGenerator().batch_to_json(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

