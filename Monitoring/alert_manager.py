from datetime import datetime

class AlertManager:
    def __init__(self, consecutive_critical_threshold=3):
        self.consecutive_critical_threshold = consecutive_critical_threshold
        self.consecutive_critical_count = 0
        self.alert_history = []

    def evaluate(self, batch_num, drift_result, perf_result):
        data_drifted = drift_result['is_drifted'] # overall drift score > 0.3
        performance_degraded = perf_result['raw_degraded'] # raw degradation diagnosis, agnostic of retraining history

        if data_drifted and performance_degraded:
            level = 'CRITICAL'
        elif data_drifted or performance_degraded:
            level = 'WARNING'
        else:
            level = 'HEALTHY'
        
        if level == 'CRITICAL':
            self.consecutive_critical_count += 1
        else:
            self.consecutive_critical_count = 0
        

        should_retrain = (
            self.consecutive_critical_count >= self.consecutive_critical_threshold
            and not perf_result['cooldown_active']
        )

        alert = {
            'timestamp': datetime.now().isoformat(),
            'batch_num' : batch_num,
            'level': level,
            'data_drifted': data_drifted,
            'drift_score': drift_result['overall_drift_score'],
            'performance_degraded': performance_degraded,
            'f1_score': perf_result['metrics']['f1'],
            'consecutive_critical_count': self.consecutive_critical_count,
            'should_retrain': should_retrain
        }

        self.alert_history.append(alert)
        return alert

    def reset_streak(self):
        self.consecutive_critical_count = 0
