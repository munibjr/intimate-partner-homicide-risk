from src.api.app import create_app
from src.data.case_generator import CaseGenerator


def test_health_endpoint():
    """Health endpoint should report API status."""
    client = create_app().test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_endpoint_valid_payload():
    """Predict endpoint should return a risk report for a valid case."""
    client = create_app().test_client()
    case = CaseGenerator(seed=42).generate_high_risk_case()

    response = client.post("/predict", json=case)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["case_id"] == case["case_id"]
    assert "disclaimer" in payload


def test_predict_endpoint_invalid_payload():
    """Predict endpoint should reject invalid payloads."""
    client = create_app().test_client()

    response = client.post("/predict", json={"case_id": "bad"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_request"


def test_batch_predict_endpoint():
    """Batch endpoint should return multiple reports."""
    client = create_app().test_client()
    cases = CaseGenerator(seed=42).generate_dataset(n_cases=3, high_risk_ratio=0.34)

    response = client.post("/batch-predict", json={"cases": cases})

    assert response.status_code == 200
    assert len(response.get_json()["predictions"]) == 3

