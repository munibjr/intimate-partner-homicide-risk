"""Middleware helpers for the Flask API."""

import logging
import os
from functools import wraps
from typing import Any, Callable

from flask import jsonify, request


logger = logging.getLogger("risk_api")


def configure_logging() -> None:
    """Configure simple request logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def require_demo_api_key(view: Callable[..., Any]) -> Callable[..., Any]:
    """Require x-api-key only when DEMO_API_KEY is configured."""

    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        expected_key = os.getenv("DEMO_API_KEY")
        if expected_key and request.headers.get("x-api-key") != expected_key:
            return jsonify({"error": "unauthorized"}), 401
        return view(*args, **kwargs)

    return wrapper


def log_request() -> None:
    """Log a compact request line."""
    logger.info("%s %s", request.method, request.path)

