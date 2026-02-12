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


### instead of hard thresholds
### find 5th percentile for last x batches
### and if current performance is less than that percentile value, then alert
### add a cooldown period after retraining (i.e. no retraining for at least the next five batches)


class PerformanceEvaluator:
    def __init__(self, performance_thresholds=None):
        self.performance_thresholds = performance_thresholds or {
            'accuracy_threshold': 0.8,
            'precision_threshold': 0.75,
            'f1_threshold': 0.8
        } ### no hard thresholds
