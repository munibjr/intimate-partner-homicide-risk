import json

from src.cli.batch_predict import main as batch_main
from src.cli.predict import main as predict_main
from src.data.case_generator import CaseGenerator


def test_predict_cli_json_output(tmp_path, capsys):
    """Single-case CLI should emit a JSON report."""
    case = CaseGenerator(seed=42).generate_high_risk_case()
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")

    exit_code = predict_main([str(path), "--format", "json"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["case_id"] == case["case_id"]
    assert "disclaimer" in output


def test_batch_predict_cli_json_output(tmp_path, capsys):
    """Batch CLI should emit a JSON prediction list."""
    cases = CaseGenerator(seed=42).generate_dataset(n_cases=4, high_risk_ratio=0.5)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")

    exit_code = batch_main([str(path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(output["predictions"]) == 4
    assert "disclaimer" in output

