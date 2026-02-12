import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime


class DriftDetector:
    def __init__(self, reference_data, features):
        self.reference_data = reference_data[features]
        self.features = features

        self.reference_stats = {
            'reference_means': self.reference_data.mean().to_dict(),
            'reference_std': self.reference_data.std().to_dict(),
            'size': len(self.reference_data)
        }
    

    def detect_drift(self, current_batch_data, threshold=0.05):
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
            reference_values = self.reference_data[feature]
            current_values = current_features[feature]

            result = stats.ks_2samp(reference_values, current_values)
            is_drifted = result.pvalue < threshold
            drift_direction = 'none'

            if is_drifted:
                drift_direction = 'increased' if result.statistic_sign == -1 else 'decreased'
                num_drifted += 1
                drift_stats['drifted_features'].append(feature)

            reference_mean = self.reference_stats['reference_means'][feature]
            current_mean = current_values.mean()
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
        pass

