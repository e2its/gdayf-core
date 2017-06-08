from gdayf.common.metricmetadata import MetricMetadata


class RegressionMetricMetadata(MetricMetadata):
    def __init__(self):
        super(RegressionMetricMetadata, self).__init__()

    def set_accuracy(self, threshold):
        None

    def set_precision(self, threshold):
        None

    def set_metrics(self, perf_metrics):
        for parameter, _ in self.items():
            try:
                self[parameter] = perf_metrics._metric_json[parameter]
            except KeyError:
                None