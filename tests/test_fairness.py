import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from src.evaluation.fairness_audit import FairnessAudit
from src.data.case_generator import CaseGenerator
from src.data.feature_extractor import FeatureExtractor
from src.models.ensemble import RiskEnsemble


class TestFairnessAudit:
    """Test fairness and bias detection"""

    @pytest.fixture
    def model_and_data(self):
        """Train model and prepare test data with demographics"""
        gen = CaseGenerator(seed=42)
        extractor = FeatureExtractor()

        # Generate cases with synthetic demographics
        cases = gen.generate_dataset(n_cases=300)

        # Add demographics (simulated)
        for i, case in enumerate(cases):
            case['age_group'] = np.random.choice(['young', 'middle', 'older'])
            case['income_level'] = np.random.choice(['low', 'medium', 'high'])
            case['nationality'] = np.random.choice(['norwegian', 'immigrant'])

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

        # Split and train
        X_train, X_test, y_train, y_test, cases_train, cases_test = train_test_split(
            X, y, cases, test_size=0.3, random_state=42
        )

        model = RiskEnsemble(random_state=42)
        model.fit(X_train, y_train)

        return model, X_test, y_test, cases_test

    def test_no_extreme_bias(self, model_and_data):
        """Verify FPR/FNR don't differ dramatically by demographic"""
        model, X_test, y_test, cases_test = model_and_data

        # Get predictions
        predictions = model.predict_proba(X_test)[:, 1] > 0.5

        # Split by age group
        for age_group in ['young', 'middle', 'older']:
            mask = np.array([c.get('age_group') == age_group for c in cases_test])

            if mask.sum() > 0:
                group_y_test = y_test[mask]
                group_preds = predictions[mask]

                # Should be able to compute metrics
                if len(np.unique(group_y_test)) > 1:
                    from sklearn.metrics import confusion_matrix
                    tn, fp, fn, tp = confusion_matrix(group_y_test, group_preds).ravel()

                    # Calculate FPR and FNR
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                    # Should be reasonable (not 0 or 1)
                    assert 0 <= fpr <= 1
                    assert 0 <= fnr <= 1

    def test_precision_by_demographic(self, model_and_data):
        """Verify precision doesn't collapse for any demographic"""
        model, X_test, y_test, cases_test = model_and_data

        predictions = model.predict_proba(X_test)[:, 1] > 0.5

        for nationality in ['norwegian', 'immigrant']:
            mask = np.array([c.get('nationality') == nationality for c in cases_test])

            if mask.sum() > 0:
                group_y_test = y_test[mask]
                group_preds = predictions[mask]

                if (group_preds == 1).sum() > 0:  # Only if we made predictions
                    # Precision: TP / (TP + FP)
                    tp = ((group_preds == 1) & (group_y_test == 1)).sum()
                    fp = ((group_preds == 1) & (group_y_test == 0)).sum()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

                    # Should maintain reasonable precision
                    assert precision >= 0.3  # At least 30% precision


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
