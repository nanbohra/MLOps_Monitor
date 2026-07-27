import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import f1_score, accuracy_score

from drift_detector import DriftDetector
from performance_eval import PerformanceEvaluator
from alert_manager import AlertManager
from retrainer import train_challenger

class Orchestrator:
    def __init__(self, config):
        self.config = config # current base model, data paths, imp stuff
        self.features = config['features']
        self.target = config['target']

        # current in-production model
        self.model = joblib.load(config['model_path'])
        self.scaler = joblib.load(config['scaler_path'])

        baseline_df = pd.read_csv(config['baseline_data_path']) # all the data accumulated so far
        self.seen_data = baseline_df

        # init monitors 
        self.drift_detector = DriftDetector(baseline_df, self.features)
        self.performance_evaluator = PerformanceEvaluator(
            window_size=config.get('window', 10),
            percentile=config.get('percentile', 5),
            min_history=config.get('min_history', 5),
            cooldown=config.get('cooldown_period',5)
        )

        # init alert manager
        self.alert_manager = AlertManager(
            consecutive_critical_threshold=config.get('consective_threshold',3)
        )

        # states, setup for shadow challengers
        self.state = 'MONITORING'
        self.shadow_challenger = None
        self.shadow_results = []
        self.shadow_target_batches = config.get('shadow_target_batches',4) # perform better on 4 shadow batches to be promoted to production

        # history
        self.batch_log = []
