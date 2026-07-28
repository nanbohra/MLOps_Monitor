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
        self.shadow_challenger = None # tuple of model, scaler
        self.shadow_results = []
        self.shadow_target_batches = config.get('shadow_target_batches',4) # perform better on 4 shadow batches to be promoted to production

        # history
        self.batch_log = []

    def process_batch(self, batch_df, batch_num):
        # records batch data into seen_data unconditionally for eventual retraining
        # triggers monitors based on state (monitoring vs. shadow)
        # sends batch records from ^ to log

        self.seen_data.append(batch_df)

        if self.state == 'MONITORING':
            record = self._run_monitoring(batch_df, batch_num)
        elif self.state == 'SHADOW_TEST':
            record = self._run_shadow_test(batch_df, batch_num)
        else:
            raise ValueError(f'Unknown state: {self.state}.')
        
        self.batch_log.append(record)
        return record

    # support functions
    def _run_monitoring(self, batch_df, batch_num):
        # feed data features into drift detector
        # feed predictions vs ground truths into performance eval
        # feed detector data into alert manager

        # collate specific information into a record for logging
        # trigger retraining if required

        X = batch_df[self.features] 
        y_true = batch_df[self.target]

        y_pred, y_probs = self._predict(self.model, self.scaler, X)

        drift_result = self.drift_detector.detect_drift(batch_df)
        perf_result = self.performance_evaluator.process_batch(
            y_true=y_true, y_pred=y_pred, y_pred_probs=y_probs, batch_num=batch_num
        )
        alert = self.alert_manager.evaluate(batch_num, drift_result, perf_result)

        # records between monitoring state and shadowtest state are not identical in structure
        # but overlapping fields to create congruency
        record = {
            'batch_num': batch_num,
            'state': 'MONITORING',
            'alert_level': alert['level'],
            'production_f1': perf_result['metrics']['f1'],
            'production_acc': perf_result['metrics']['accuracy'],
            'drift_score': drift_result['overall_drift_score'],
            'should_retrain': alert['should_retrain']
        }

        print(f"Batch {batch_num:02d} [MONITORING]: {alert['level']} | "
              f"F1={perf_result['metrics']['f1']:.3f} | "
              f"drift={drift_result['overall_drift_score']:.1%}")
        
        if alert['should_retrain']:
            self._start_retrain(batch_num)

        return record


    def _predict(self, model, scaler, X):
        # useful for shadow testing with parameterized model pieces
        # run inference on both models concurrently

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_probs = model.predict_proba(X_scaled)[:, 1]
        return y_pred, y_probs
        
    
    def _start_retrain(self, batch_num):
        # collates training data to send to retrainer
        # trigger retraining with imported function
        # handles state change
        print(f"------ Retrain triggered at batch {batch_num}. Training challenger.")

        train_data = self._all_seen_data()
        challenger_model, challenger_scaler = train_challenger(
            training_data=train_data,
            features=self.features,
            target=self.target
        )

        self.shadow_challenger = (challenger_model,challenger_scaler)
        self.shadow_results = []
        self.state = 'SHADOW_TEST'

        print(f"------ Challenger trained on {len(train_data)} samples. "
              f"Entering SHADOW TEST for next {self.shadow_target_batches} batches.")
        
    def _all_seen_data(self):
        # concat all batch data in seen_data list for eventual retraining
        return pd.concat(self.seen_data, ignore_index=False)
    
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
    



    
