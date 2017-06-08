from gdayf.common.metricmetadata import MetricMetadata
from collections import OrderedDict


class MultinomialMetricMetadata(MetricMetadata):
    def __init__(self):
        super(MultinomialMetricMetadata, self).__init__()
        self['hit_ratio_table'] = OrderedDict()
        self['cm'] = OrderedDict()

    def set_accuracy(self, threshold):
        None

    def set_precision(self, threshold):
        None
