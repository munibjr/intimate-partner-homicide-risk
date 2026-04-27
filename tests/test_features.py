import pytest
import numpy as np
from src.data.case_generator import CaseGenerator
from src.data.feature_extractor import FeatureExtractor


class TestFeatureExtraction:
    """Detailed feature extraction tests"""

    def test_violence_escalation_detection(self):
        """High-risk cases should show violence escalation"""
        gen = CaseGenerator(seed=42)
        extractor = FeatureExtractor()

        high_risk = gen.generate_high_risk_case()
        features = extractor.extract_features(high_risk)

        # High-risk should have detectable escalation
        assert features['violence_escalation_rate'] > 0.2

    def test_threat_frequency_high_risk(self):
        """High-risk cases should have higher threat frequencies"""
        gen = CaseGenerator(seed=42)
        extractor = FeatureExtractor()

        # Generate multiple high-risk and low-risk cases
        high_risk_threats = []
        low_risk_threats = []

        for _ in range(10):
            high = extractor.extract_features(gen.generate_high_risk_case())
            low = extractor.extract_features(gen.generate_low_risk_case())

            high_risk_threats.append(high['threat_frequency'])
            low_risk_threats.append(low['threat_frequency'])

        # Average threat frequency should be higher in high-risk
        assert np.mean(high_risk_threats) > np.mean(low_risk_threats)

    def test_police_contact_frequency(self):
        """Verify police contact frequency is tracked"""
        gen = CaseGenerator(seed=42)
        extractor = FeatureExtractor()

        case = gen.generate_high_risk_case()
        features = extractor.extract_features(case)

        assert 'police_contact_frequency' in features
        assert 0 <= features['police_contact_frequency'] <= 1

    def test_multi_agency_involvement(self):
        """Verify detection of multi-agency involvement"""
        gen = CaseGenerator(seed=42)
        extractor = FeatureExtractor()

        case = gen.generate_high_risk_case()
        features = extractor.extract_features(case)

        # High-risk should involve multiple agencies
        assert features['multiple_agencies_involved'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
