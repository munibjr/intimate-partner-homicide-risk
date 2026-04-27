"""Flask application factory for the demo API."""

from flask import Flask

from src.api.middleware import configure_logging, log_request
from src.api.routes import api
from src.models.service import RiskPredictionService


def create_app() -> Flask:
    """Create and configure the Flask app."""
    configure_logging()
    app = Flask(__name__)
    app.config["prediction_service"] = RiskPredictionService()
    app.before_request(log_request)
    app.register_blueprint(api)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

