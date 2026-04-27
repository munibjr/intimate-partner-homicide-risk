"""
Fairness auditing framework for bias detection
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from typing import Dict, List


class FairnessAudit:
    """Comprehensive fairness analysis"""

    def __init__(self):
        self.results = {}

    def audit_demographic_parity(self, predictions, demographics, feature_name):
        """
        Check if positive prediction rate differs by demographic
        """
        results = {}

        for group_value in np.unique(demographics):
            mask = demographics == group_value

            positive_rate = (predictions[mask] == 1).sum() / mask.sum() if mask.sum() > 0 else 0
            results[group_value] = {
                'positive_rate': float(positive_rate),
                'n_samples': int(mask.sum())
            }

        # Check for significant disparity
        positive_rates = [r['positive_rate'] for r in results.values() if isinstance(r, dict)]
        if len(positive_rates) > 1:
            max_disparity = max(positive_rates) - min(positive_rates)
            results['max_disparity'] = float(max_disparity)
            results['has_disparity'] = max_disparity > 0.15  # Flag if >15% difference

        self.results[f'demographic_parity_{feature_name}'] = results
        return results

    def audit_equalized_odds(self, predictions, demographics, labels, feature_name):
        """
        Check if TPR and FPR are equal across demographics
        """
        results = {}

        for group_value in np.unique(demographics):
            mask = demographics == group_value

            group_labels = labels[mask]
            group_preds = predictions[mask]

            if len(np.unique(group_labels)) < 2:  # Need both classes
                continue

            if len(group_labels) > 0:
                tn, fp, fn, tp = confusion_matrix(group_labels, group_preds).ravel()

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                results[group_value] = {
                    'tpr': float(tpr),
                    'fpr': float(fpr),
                    'n_samples': int(mask.sum())
                }

        # Check for disparity
        if len(results) > 1:
            tprs = [r['tpr'] for r in results.values() if isinstance(r, dict) and 'tpr' in r]
            fprs = [r['fpr'] for r in results.values() if isinstance(r, dict) and 'fpr' in r]

            if tprs:
                results['tpr_disparity'] = float(max(tprs) - min(tprs))
            if fprs:
                results['fpr_disparity'] = float(max(fprs) - min(fprs))

        self.results[f'equalized_odds_{feature_name}'] = results
        return results

    def audit_calibration(self, predicted_probs, demographics, labels, feature_name):
        """
        Check if predicted probabilities are calibrated by demographic
        """
        results = {}

        for group_value in np.unique(demographics):
            mask = demographics == group_value

            group_labels = labels[mask]
            group_probs = predicted_probs[mask]

            if len(group_labels) > 0:
                # Calibration: compare predicted prob to actual positive rate
                actual_positive_rate = group_labels.mean()
                avg_predicted_prob = group_probs.mean()

                calibration_gap = abs(actual_positive_rate - avg_predicted_prob)

                results[group_value] = {
                    'actual_positive_rate': float(actual_positive_rate),
                    'avg_predicted_prob': float(avg_predicted_prob),
                    'calibration_gap': float(calibration_gap),
                    'n_samples': int(mask.sum())
                }

        self.results[f'calibration_{feature_name}'] = results
        return results

    def print_audit_summary(self):
        """Print fairness audit results"""
        print("\n" + "="*60)
        print("FAIRNESS AUDIT SUMMARY")
        print("="*60)

        for audit_name, results in self.results.items():
            print(f"\n{audit_name.upper()}")
            print("-" * 40)

            for key, value in results.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subval in value.items():
                        if isinstance(subval, float):
                            print(f"    {subkey}: {subval:.4f}")
                        else:
                            print(f"    {subkey}: {subval}")
                else:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    elif isinstance(value, bool):
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {value}")

        print("\n" + "="*60)
