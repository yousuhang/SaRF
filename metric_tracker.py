import numpy as np
class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def moving_average(self, old, new):
        s = 0.98
        return old * (s) + new * (1 - s)

    def update_metrics(self, metric_dict, smoothe=True):
        for k, v in metric_dict.items():
            if k in self.metrics.keys():
                self.metrics[k] = np.concatenate((self.metrics[k],np.expand_dims(v, axis=0)))
            else:
                self.metrics[k] = np.expand_dims(v, axis=0)

    def current_metrics(self):
        return self.metrics
