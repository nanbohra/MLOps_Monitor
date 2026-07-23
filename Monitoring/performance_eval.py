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
        