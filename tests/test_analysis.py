from src.analysis.pattern_detector import PatternDetector
from src.analysis.risk_trajectory import RiskTrajectory
from src.analysis.trend_analyzer import TrendAnalyzer
from src.data.case_generator import CaseGenerator


def test_trend_analyzer_detects_worsening_case():
    """High-risk synthetic cases should show a worsening trend."""
    case = CaseGenerator(seed=42).generate_high_risk_case()

    result = TrendAnalyzer().analyze(case)

    assert result["trend"] == "worsening"


def test_risk_trajectory_returns_one_point_per_event():
    """Trajectory should include cumulative points for every event."""
    case = CaseGenerator(seed=42).generate_high_risk_case()

    trajectory = RiskTrajectory().build(case)

    assert len(trajectory) == len(case["events"])
    assert trajectory[-1]["trajectory_score"] >= trajectory[0]["trajectory_score"]


def test_pattern_detector_finds_protection_breach():
    """Pattern detector should identify protective order breaches."""
    case = CaseGenerator(seed=42).generate_high_risk_case()

    patterns = PatternDetector().detect(case)

    assert "protective_order_breach" in {pattern["pattern"] for pattern in patterns}

