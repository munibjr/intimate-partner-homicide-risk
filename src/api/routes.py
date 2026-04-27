"""Flask routes for risk predictions."""

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from src.api.middleware import require_demo_api_key
from src.api.models import BatchPredictionRequest, CaseRequest
from src.models.service import DISCLAIMER, RiskPredictionService


api = Blueprint("api", __name__)


def get_service() -> RiskPredictionService:
    """Get the app-scoped prediction service."""
    return current_app.config["prediction_service"]


@api.get("/health")
def health() -> tuple[dict, int]:
    """Return API health metadata."""
    return get_service().health(), 200


@api.post("/predict")
@require_demo_api_key
def predict() -> tuple[dict, int]:
    """Predict risk for a single synthetic case."""
    try:
        payload = CaseRequest.model_validate(request.get_json(force=True))
    except ValidationError as exc:
        return jsonify({"error": "invalid_request", "details": exc.errors()}), 400

    report = get_service().predict_case(payload.to_case())
    return jsonify(report), 200


@api.post("/batch-predict")
@require_demo_api_key
def batch_predict() -> tuple[dict, int]:
    """Predict risk for multiple synthetic cases."""
    try:
        payload = BatchPredictionRequest.model_validate(request.get_json(force=True))
    except ValidationError as exc:
        return jsonify({"error": "invalid_request", "details": exc.errors()}), 400

    reports = get_service().predict_batch([case.to_case() for case in payload.cases])
    return jsonify({"predictions": reports, "disclaimer": DISCLAIMER}), 200

