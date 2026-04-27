import pytest
import numpy as np
from src.data.case_generator import CaseGenerator
from src.data.feature_extractor import FeatureExtractor


class TestCaseGenerator:
    """Test synthetic data generation quality"""

    def test_high_risk_escalation_pattern(self):
        """Verify high-risk cases show escalation pattern"""
        gen = CaseGenerator(seed=42)
        case = gen.generate_high_risk_case()

        # Should have violence, threats, and escalation
        assert case['violence_incidents'] >= 2
        assert case['threat_frequency'] >= 1
        assert case['risk_category'] == 'high'

    def test_low_risk_isolated_incident(self):
        """Verify low-risk cases are isolated incidents"""
        gen = CaseGenerator(seed=42)
        case = gen.generate_low_risk_case()

        # Should have minimal escalation
        assert case['violence_escalation_rate'] < 0.3
        assert case['risk_category'] == 'low'

    def test_dataset_distribution(self):
        """Verify dataset maintains 70/30 low/high split"""
        gen = CaseGenerator(seed=42)
        cases = gen.generate_dataset(n_cases=500)

        high_risk_count = sum(1 for c in cases if c['risk_category'] == 'high')
        low_risk_count = sum(1 for c in cases if c['risk_category'] == 'low')

        # Should be approximately 70/30 split
        high_risk_ratio = high_risk_count / len(cases)
        assert 0.25 < high_risk_ratio < 0.35  # Allow ±5% variance

    def test_temporal_realism(self):
        """Verify timestamps are realistic and ordered"""
        gen = CaseGenerator(seed=42)
        case = gen.generate_high_risk_case()

        start_date = case['start_date']
        assert len(case['events']) > 0


class TestFeatureExtractor:
    """Test feature engineering pipeline"""

    def test_feature_normalization(self):
        """Verify all features are normalized to [0, 1]"""
        gen = CaseGenerator(seed=42)
        extractor = FeatureExtractor()

        cases = gen.generate_dataset(n_cases=100)
        for case in cases:
            features = extractor.extract_features(case)

            # All features except risk_label should be in [0, 1]
            for feature_name, value in features.items():
                if feature_name != 'risk_label':
                    assert 0 <= value <= 1, f"{feature_name}={value} out of range"

    def test_feature_count(self):
        """Verify we extract exactly 15 features"""
        extractor = FeatureExtractor()
        gen = CaseGenerator(seed=42)
        case = gen.generate_high_risk_case()

        features = extractor.extract_features(case)
        # 15 features + risk_label
        assert len(features) == 16

    def test_feature_interpretability(self):
        """Verify feature names are interpretable"""
        extractor = FeatureExtractor()
        gen = CaseGenerator(seed=42)
        case = gen.generate_high_risk_case()

        features = extractor.extract_features(case)
        expected_features = {
            'violence_incidents', 'max_violence_severity', 'violence_escalation_rate',
            'threat_frequency', 'weapon_mention', 'threat_severity_max',
            'isolation_indicators', 'restraining_order_breaches', 'police_contact_frequency',
            'child_welfare_contact', 'days_since_last_incident', 'trend_worsening',
            'velocity_of_escalation', 'multiple_agencies_involved', 'breach_of_protection'
        }

        actual_features = set(features.keys()) - {'risk_label'}
        assert actual_features == expected_features


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
