import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime


class DriftDetector:
    def __init__(self, reference_data, features):
        self.reference_data = reference_data[features]
        self.features = features

        self.reference_stats = { # create baseline stats
            'reference_means': self.reference_data.mean().to_dict(),
            'reference_std': self.reference_data.std().to_dict(),
            'size': len(self.reference_data)
        }

        print(f"Drift detector initialized with {self.reference_stats['size']} reference samples.")
    

    def detect_drift(self, current_batch_data, threshold=0.05):
        # centered on KS test of distributions, 2sample to check if data comes from the same dist
        current_features = current_batch_data[self.features]

        drift_stats = {
            'timestamp' : datetime.now().isoformat(),
            'batch_size': len(current_features),
            'feature_stats' : {},
            'overall_drift_score': 0.0,
            'is_drifted': False,
            'drifted_features': []
        }

        num_drifted = 0

        for feature in self.features:
            # for each feature, compare curr batch data against reference batch data
            reference_values = self.reference_data[feature]
            current_values = current_features[feature]

            result = stats.ks_2samp(reference_values, current_values)
            is_drifted = result.pvalue < threshold

            if is_drifted:
                num_drifted += 1
                drift_stats['drifted_features'].append(feature)

            reference_mean = self.reference_stats['reference_means'][feature]
            current_mean = current_values.mean()
            drift_direction = 'increased' if current_mean > reference_mean else 'decreased'
            mean_shift = ((current_mean - reference_mean) / reference_mean) if reference_mean != 0 else 0

            drift_stats['feature_stats'][feature] = {
                'ks_statistic' : float(result.statistic),
                'p_value' : float(result.pvalue),
                'is_drifted': bool(is_drifted),
                'drift_direction': drift_direction,
                'current_batch_mean': float(current_mean),
                'mean_shift': float(mean_shift * 100.0)
            }

        
        drift_stats['overall_drift_score'] = num_drifted / len(self.features)
        drift_stats['is_drifted'] = drift_stats['overall_drift_score'] > 0.3

        return drift_stats
    

    def detect_pred_drift():
        # TODO
        # detecting whether prediction probability distributions have shifted
        # i.e. if feature drifts have had meaningful downstream effect on label distributions
        # OR if label probabilities have changed despite no feature drift (rare)
        # re-evaluate its need here
        # doesn't provide any independent signal for retraining necessity beyond minor additional filter
        # i.e. if feature drift detected but no change in pred probs -- is data drift really sig enough to warrant retraining?
        pass

    def update_reference(self, new_reference_data):
        # after retraining a model, old reference data distributions are outdated
        # so drifts may be flagged relative to old distributions even if retrained model 
        # is operating on new distributions since retraining
        # so we update reference distributions to more accurately detect drift after retrains

        # mirrors PerformanceEvaluator's notify_retrain()

        self.reference_data = new_reference_data[self.features]
        self.reference_stats = {
            'reference_means' : self.reference_data.mean().to_dict(),
            'reference_std' : self.reference_data.std().to_dict(),
            'size': len(self.reference_data)
        }

        print(f"Drift detector reference updated with {self.reference_stats['size']} samples.")


