"""
create a monitor that watches performance and checks against thresholds with every batch
then send alerts based on performance metrics dipping below thresholds 
based on alert severity and number of consecutive warnings, trigger retraining
have a main type file that manages simultaneous monitoring of feature drift and performance deterioriation
trigger retraining based on alerts
separate file for retraining

"""

## Basically a objectified version of run_model_drift.ipynb
## 'Streams' live data, performs classification, collects metrics
## Watches for when these metrics dip below set thresholds


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from collections import deque
from datetime import datetime


### instead of hard thresholds
### find 5th percentile for last x batches
### and if current performance is less than that percentile value, then alert
### add a cooldown period after retraining (i.e. no retraining for at least the next five batches)

# TODO
## calculate metrics of model performance on current batch
## maintain rolling window of performance metrics on past batches
## compute nth percentile (?) of each window as 
## compare current metrics against above and flag the need for retraining
## handle warm-up -- what to do when there isn't enough history of batches to create a valid threshold
## handle cool-down -- don't flag for retraining when recently retrained
## reset the history of performance metrics after model change / retrain

class PerformanceEvaluator:
    def __init__(self, track_metrics=None, window_size=10, percentile=5, min_history=5, cooldown=5):
        self.track_metrics = track_metrics or ['accuracy', 'precision', 'f1', 'recall', 'roc_auc']
        self.window_size = window_size
        self.percentile = percentile
        self.min_history = min_history
        self.cooldown = cooldown

        self.history = deque(maxlen=window_size) # maintains rolling window of history to track metrics from
        self.last_retrain_batch_num = None


    def evaluate_batch(self, y_true, y_pred, y_pred_probs):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_probs)
        }
        
    def get_thresholds(self):
        if len(self.history) < self.min_history:
            return None # if we dont have enough data yet, no thresholds for metrics yet
        
        return {
            metric : float(np.percentile([b[metric] for b in self.history], self.percentile))
            for metric in self.track_metrics
        }
        
    def check_degradation(self, metrics):
        # check if inf on current batch performance significantly underperforms 
        thresholds = self.get_thresholds()

        if thresholds is None:
            return False, [], None
        
        failed_metrics = [
            {'metric' : m, 'value': metrics[m], 'threshold': thresholds[m], 'gap': metrics[m] - thresholds[m]}
            for m in self.track_metrics if metrics[m] < thresholds[m]
        ]

        return len(failed_metrics) > 0, failed_metrics, thresholds
        

    def in_cooldown(self, current_batch_num):
        if self.last_retrain_batch_num is None: # if retraining never happened
            return False # then can't be in cooldown
        
        return (current_batch_num - self.last_retrain_batch_num) < self.cooldown
    

    def process_batch(self, y_true, y_pred, y_pred_probs, batch_num):
        metrics = self.evaluate_batch(y_true, y_pred,y_pred_probs)
        is_degraded, degraded_metrics, thresholds_used = self.check_degradation(metrics)

        cooldown_active = self.in_cooldown(batch_num)
        retraining_required = is_degraded and not cooldown_active

        self.history.append(metrics) # check degradation first, then add batch metrics to history
                                        # to avoid artificially deflating / inflating history metrics for threshold calc

        return {
            'timestamp' : datetime.now().isoformat(),
            'batch_num' : batch_num,
            'metrics': metrics,
            'retraining_required' : retraining_required,
            'raw_degraded' : is_degraded,
            'cooldown_active': cooldown_active,
            'degraded_metrics': degraded_metrics,
            'thresholds_used': thresholds_used,
            'history_size': len(self.history)
        }

    def notify_retrain(self, batch_num):
        self.last_retrain_batch_num = batch_num
        self.history.clear()

