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
        self.seen_data = [baseline_df]

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

    def process_batch():
        # TODO
        # records batch data into seen_data unconditionally for eventual retraining
        # triggers monitors based on state (monitoring vs. shadow)
        # sends batch records from ^ to log

    # support functions
    def _run_monitoring():
        # TODO
        # feed data features into drift detector
        # feed predictions vs ground truths into performance eval
        # feed detector data into alert manager

        # collate specific information into a record for logging
        # trigger retraining if required


    def _predict():
        # TODO
        # useful for shadow testing with parameterized model pieces
        # run inference on both models concurrently

    
    def _start_retrain():
        # TODO
        # collates training data to send to retrainer
        # trigger retraining with imported function
        # handles state change
    
    def _all_seen_data():
        # TODO
        # concat all batch data in seen_data list for eventual retraining
    
    def _run_shadow_test():
        # TODO
        # collect feature data + ground truths 
        # run drift detection on data, but not passed to alert manager, since alerts are paused during shadow testing
        # run both models through prediction / inference
        # record performance of production model (performanceEva)
        # get all metrics on challenger model also
        # create logs for all sensors
        # if we've run teh challenger for enough test batches, send to promotion evaluation

    def _evaluate_promotion():
        # TODO
        # evaluate metrics across shadow testing to see which model prevails
        # CURRENT CRITERION FOR PROMOTION: improved avg f1
        # check against config threshold if avg f1 has improved 
        # replace production model with retrained model
        # update reference data, notify retrain on performanceEval
        # reset alert manager
        # reset initializations for shadow challenger
        # reset state
        # ELSE discard challenger if it didn't outperform
            # retraining will be re-attempted after cooldown

    def save_log():
        # TODO
        # save logs to CSV
    
    


    
