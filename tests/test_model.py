import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.case_generator import CaseGenerator
from src.data.feature_extractor import FeatureExtractor
from src.models.ensemble import RiskEnsemble
from src.evaluation.metrics import ModelEvaluator


class TestRiskEnsemble:
    """Test ensemble model training and prediction"""

    @pytest.fixture
    def train_data(self):
        """Generate training data"""
        gen = CaseGenerator(seed=42)
        extractor = FeatureExtractor()

        cases = gen.generate_dataset(n_cases=200)
        features_list = [extractor.extract_features(c) for c in cases]

        feature_names = [
            'violence_incidents', 'max_violence_severity', 'violence_escalation_rate',
            'threat_frequency', 'weapon_mention', 'threat_severity_max',
            'isolation_indicators', 'restraining_order_breaches', 'police_contact_frequency',
            'child_welfare_contact', 'days_since_last_incident', 'trend_worsening',
            'velocity_of_escalation', 'multiple_agencies_involved', 'breach_of_protection'
        ]

        X = np.array([[f[feat] for feat in feature_names] for f in features_list])
        y = np.array([f['risk_label'] for f in features_list])

        return X, y

    def test_model_training(self, train_data):
        """Verify model trains without errors"""
        X, y = train_data
        model = RiskEnsemble(random_state=42)

        # Should train without raising
        model.fit(X, y)
        assert model.is_trained == True

    def test_prediction_output(self, train_data):
        """Verify predictions are valid probabilities"""
        X, y = train_data
        model = RiskEnsemble(random_state=42)
        model.fit(X, y)

        # Make predictions
        probs = model.predict_proba(X)

        # Check probability constraints
        assert probs.shape == (len(X), 2)
        assert np.all((probs >= 0) & (probs <= 1))
        assert np.allclose(probs.sum(axis=1), 1.0)  # Sum to 1

    def test_risk_scores(self, train_data):
        """Verify risk scores are 0-100"""
        X, y = train_data
        model = RiskEnsemble(random_state=42)
        model.fit(X, y)

        scores = model.get_risk_scores(X)

        assert np.all((scores >= 0) & (scores <= 100))
        assert len(scores) == len(X)

    def test_cross_validation_scoring(self, train_data):
        """Verify cross-validation produces reasonable scores"""
        X, y = train_data
        model = RiskEnsemble(random_state=42)

        # Perform training (includes CV)
        cv_results = model.fit(X, y)

        # Should have decent performance (not random)
        assert cv_results['cv_mean'] > 0.6  # Better than random


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
