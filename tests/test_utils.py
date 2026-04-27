import json

from src.data.case_generator import CaseGenerator
from src.models.service import DISCLAIMER, RiskPredictionService
from src.utils.case_loader import CaseLoader
from src.utils.report_generator import ReportGenerator


def test_json_case_loader_single_case(tmp_path):
    """JSON loader should accept a single case object."""
    case = CaseGenerator(seed=42).generate_high_risk_case()
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")

    loaded = CaseLoader().load(str(path))

    assert len(loaded) == 1
    assert loaded[0]["case_id"] == case["case_id"]


def test_csv_case_loader_events_column(tmp_path):
    """CSV loader should parse an events JSON column."""
    case = CaseGenerator(seed=42).generate_low_risk_case()
    path = tmp_path / "cases.csv"
    events_json = json.dumps(case["events"]).replace('"', '""')
    path.write_text(
        f'case_id,start_date,events,risk_category\nCSV_001,{case["start_date"]},"{events_json}",low\n',
        encoding="utf-8",
    )

    loaded = CaseLoader().load(str(path))

    assert loaded[0]["case_id"] == "CSV_001"
    assert loaded[0]["events"][0]["type"] == "police_report"


def test_report_includes_disclaimer():
    """Reports should include decision-support language."""
    service = RiskPredictionService(n_cases=80)
    report = service.predict_case(CaseGenerator(seed=42).generate_high_risk_case())
    text = ReportGenerator().to_text(report)

    assert DISCLAIMER in text
    assert "Risk level:" in text

