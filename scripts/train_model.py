"""Train the synthetic risk model and print cross-validation metrics."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.service import RiskPredictionService


def main() -> int:
    """Run model training from config."""
    with open("config/model_config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    training = config["training"]
    service = RiskPredictionService(
        n_cases=training["n_cases"],
        high_risk_ratio=training["high_risk_ratio"],
        seed=training["seed"],
    )
    metrics = service.train()
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
