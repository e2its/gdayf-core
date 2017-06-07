from gdayf.common.metricmetadata import MetricMetadata
from collections import OrderedDict


class BinomialMetricMetadata(MetricMetadata):
    def __init__(self):
        super().__init__()
        self['AUC'] = None
        self['gains_lift_table'] = OrderedDict()
        self['Gini'] = None
        self['mean_per_class_error'] = None
        self['logloss'] = None
        self['max_criteria_and_metric_scores'] = OrderedDict
        self['cm'] = OrderedDict()
        self['cm']['min_per_class_accuracy'] = OrderedDict()
        self['cm']['absolute_mcc'] = OrderedDict()
        self['cm']['precision'] = OrderedDict()
        self['cm']['accuracy'] = OrderedDict()
        self['cm']['f0point5'] = OrderedDict()
        self['cm']['f2'] = OrderedDict()
        self['cm']['f1'] = OrderedDict()
        self['cm']['mean_per_class_accuracy'] = OrderedDict()
        self['thresholds_and_metric_scores'] = OrderedDict()

    def set_accuracy(self, threshold):
        None

    def set_precision(self, threshold):
        None

