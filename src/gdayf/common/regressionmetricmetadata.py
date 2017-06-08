from gdayf.common.metricmetadata import MetricMetadata


class RegressionMetricMetadata(MetricMetadata):
    def __init__(self):
        super(RegressionMetricMetadata, self).__init__()

    def set_accuracy(self, threshold):
        None

    def set_precision(self, threshold):
        None