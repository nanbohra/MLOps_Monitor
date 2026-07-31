import pandas as pd
import os
from orchestrator import Orchestrator

config = {
    'model_path': 'ChurnModel/models/churn_model_logreg.joblib',
    'scaler_path': 'ChurnModel/models/scaler_logreg.joblib',
    'baseline_data_path': 'ChurnModel/data/baseline_train.csv',
    'features': ['account_age', 'total_purchases', 'avg_purchase_value',
                 'days_since_purchase', 'login_weekly_frequency',
                 'complaints_raised', 'avg_product_rating_5', 'email_open_rate'],
    'target': 'churned',
    'consecutive_threshold': 3,
    'shadow_target_batches': 4,
    'promotion_f1_improvement': 0.02,
    'recency_decay': 0.85
}

orch = Orchestrator(config)

batch_files = sorted(f for f in os.listdir('ChurnModel/data/streaming') if f.endswith('.csv'))
for i, fname in enumerate(batch_files):
    batch = pd.read_csv(f'ChurnModel/data/streaming/{fname}')
    orch.process_batch(batch, batch_num=i)

orch.save_log('monitoring_log.csv')

# # test_recency_weights.py
# import pandas as pd
# import os
# from orchestrator import Orchestrator

# config = {
#     'model_path': 'ChurnModel/models/churn_model_logreg.joblib',
#     'scaler_path': 'ChurnModel/models/scaler_logreg.joblib',
#     'baseline_data_path': 'ChurnModel/data/baseline_train.csv',
#     'features': ['account_age', 'total_purchases', 'avg_purchase_value',
#                  'days_since_purchase', 'login_weekly_frequency',
#                  'complaints_raised', 'avg_product_rating_5', 'email_open_rate'],
#     'target': 'churned',
#     'consecutive_threshold': 3,
#     'shadow_target_batches': 4,
#     'promotion_f1_improvement': 0.02,
#     'recency_decay': 0.85
# }

# orch = Orchestrator(config)

# # feed it just enough batches to accumulate real data (don't need all 40 for this check)
# batch_files = sorted(f for f in os.listdir('ChurnModel/data/streaming') if f.endswith('.csv'))[:19]  # through batch 18
# for i, fname in enumerate(batch_files):
#     batch = pd.read_csv(f'ChurnModel/data/streaming/{fname}')
#     orch.seen_data.append(batch)   # bypass process_batch — just building up seen_data directly for this check

# # now test _all_seen_data() and the weighting logic directly, no need to trigger a real retrain
# combined = orch._all_seen_data()

# print("Shape:", combined.shape)
# print("\nNaN count in batch_num (should equal baseline row count):")
# print(combined['batch_num'].isna().sum())
# print("\nbatch_num describe:")
# print(combined['batch_num'].describe())

# # now test the weight computation itself
# weights = orch._compute_recency_weights(combined, decay=config['recency_decay'])
# print("\nWeights shape:", weights.shape)
# print("Weight range:", weights.min(), "to", weights.max())

# # sanity check: baseline rows (batch_num NaN, treated as -1) should have the smallest weights
# combined_with_weights = combined.copy()
# combined_with_weights['weight'] = weights
# print("\nAvg weight by batch_num (baseline shows as NaN):")
# print(combined_with_weights.groupby('batch_num', dropna=False)['weight'].mean())

# total_baseline_weight = combined_with_weights[combined_with_weights['batch_num'].isna()]['weight'].sum()
# total_streamed_weight = combined_with_weights[combined_with_weights['batch_num'].notna()]['weight'].sum()

# print(f"\nBaseline: {10000} rows, total weight mass = {total_baseline_weight:.1f}")
# print(f"Streamed: {1900} rows, total weight mass = {total_streamed_weight:.1f}")
# print(f"Baseline share of total fitting influence: {total_baseline_weight / (total_baseline_weight + total_streamed_weight):.1%}")