"""Evaluate the synthetic risk model on a holdout split."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.case_generator import CaseGenerator
from src.data.feature_extractor import FEATURE_NAMES, FeatureExtractor
from src.evaluation.metrics import ModelEvaluator
from src.models.ensemble import RiskEnsemble


def main() -> int:
    """Run a deterministic holdout evaluation."""
    generator = CaseGenerator(seed=42)
    extractor = FeatureExtractor()
    cases = generator.generate_dataset(n_cases=500)
    rows = [extractor.extract_features(case) for case in cases]
    x_data = np.array([[row[name] for name in FEATURE_NAMES] for row in rows])
    y_data = np.array([row["risk_label"] for row in rows])

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.25,
        random_state=42,
        stratify=y_data,
    )
    model = RiskEnsemble(random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    metrics = ModelEvaluator.evaluate(y_test, predictions, probabilities)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
