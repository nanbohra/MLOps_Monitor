import numpy as np
from performance_eval import PerformanceEvaluator

np.random.seed(42)
evaluator = PerformanceEvaluator(window_size=10, 
                                 percentile=5, 
                                 min_history=5, 
                                 cooldown=5)

def fake_batch(n=100, quality='good'):
    y_true = np.random.binomial(1, 0.3, n)
    if quality == 'good':
        # predictions mostly correct
        y_proba = np.clip(y_true + np.random.normal(0, 0.15, n), 0, 1)
    else:
        # predictions basically random / wrong
        y_proba = np.random.uniform(0, 1, n)
    y_pred = (y_proba > 0.5).astype(int)
    return y_true, y_pred, y_proba


already_retrained = False

# Simulate: batches 0-9 good, batches 10+ bad
for batch_num in range(20):
    quality = 'good' if batch_num < 10 else 'bad'
    y_true, y_pred, y_proba = fake_batch(quality=quality)
    
    result = evaluator.process_batch(y_true, y_pred, y_proba, batch_num)
    
    print(f"Batch {batch_num:2d} [{quality}]: "
          f"F1={result['metrics']['f1']:.3f}, "
          f"degraded={result['retraining_required']}, "
          f"cooldown={result['cooldown_active']}, "
          f"hist_size={result['history_size']}")
    

    # simulate retraining to check reset of history, etc.
    if result['retraining_required'] and not already_retrained:
        print(f"     simulating retraining at batch {batch_num}")
        evaluator.notify_retrain(batch_num)
        already_retrained=True