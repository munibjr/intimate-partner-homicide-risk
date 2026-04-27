"""Run a single-case risk prediction from the command line."""

import argparse
from typing import Optional

from src.models.service import RiskPredictionService
from src.utils.case_loader import CaseLoader
from src.utils.report_generator import ReportGenerator


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Predict review-priority risk for one synthetic case.")
    parser.add_argument("case_file", help="Path to a JSON file containing one case")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the single-case prediction CLI."""
    args = build_parser().parse_args(argv)
    cases = CaseLoader().load(args.case_file)
    if len(cases) != 1:
        raise ValueError("predict expects exactly one case; use batch_predict for multiple cases")

    service = RiskPredictionService()
    report = service.predict_case(cases[0])
    formatter = ReportGenerator()
    if args.format == "json":
        print(formatter.to_json(report))
    else:
        print(formatter.to_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

