import os
from churn_data import ChurnDataGenerator, generate_drift_stream


generator = ChurnDataGenerator()

baseline_data = generator.generate_batch(num_customers=10000, base_churn_rate=0.4)
baseline_data = generator.add_y_noise(baseline_data, noise_rate=0.25)
baseline_data.to_csv('baseline_train.csv', index=False)

print(f"Total customers: {len(baseline_data)}")
print(f"Churn rate: {baseline_data['churned'].mean():.2%}")
print(f"\nFeature means for churned vs non-churned:")
print(baseline_data.groupby('churned')[['days_since_purchase', 'email_open_rate']].mean())

batches = generate_drift_stream(
    generator=generator,
    magnitude=0.7,
    drift_starts=5,
    num_batches=20,
    cust_per_batch=100
)

os.makedirs('data/streaming', exist_ok=True)
for i, batch in enumerate(batches):
    batch.to_csv(f'data/streaming/batch_{i:03d}.csv', index=False)

print(f"\nGenerated {len(batches)} streaming batches")

