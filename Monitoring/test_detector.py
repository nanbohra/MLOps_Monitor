from drift_detector import DriftDetector 
import pandas as pd

baseline_data = pd.read_csv("MLOps_Monitor/ChurnModel/data/baseline_train.csv")
features = [
    'account_age','total_purchases', 'avg_purchase_value',
    'days_since_purchase', 'login_weekly_frequency',
    'complaints_raised','avg_product_rating_5','email_open_rate'
    ]

detector = DriftDetector(baseline_data, features)
detector_results = []

for i in range(20):
    batch_data = pd.read_csv(f"MLOps_Monitor/ChurnModel/data/streaming/batch_{i:03d}.csv")

    drift_result = detector.detect_drift(batch_data, threshold=0.05)

    detector_results.append({
        'batch_num': i,
        'drift_magnitude': batch_data['drift_magnitude'].iloc[0],
        'is_drifted': drift_result['is_drifted'],
        'overall_drift_score': drift_result['overall_drift_score'],
        'num_drift_features': len(drift_result['drifted_features']),
        'drifted_features': drift_result['drifted_features']
    })

    status = "DRIFTED!" if drift_result['is_drifted'] else "OK!"
    print(f"Batch {i:03d} with Drift Magnitude = {batch_data['drift_magnitude'].iloc[0]:.3f}: {status} with {len(drift_result['drifted_features'])} drifted features and drift score {drift_result['overall_drift_score']:.2%}")

drift_df = pd.DataFrame(detector_results)
drift_df.to_csv('drift_detection_results.csv')