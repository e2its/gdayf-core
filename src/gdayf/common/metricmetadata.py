from collections import OrderedDict


class MetricMetadata(OrderedDict):
    def __init__(self):
        super().__init__()
        self['MSE'] = None
        self['mean_residual_deviance'] = None
        self['nobs'] = None
        self['model_category'] = None
        self['predictions']
        self['rmsle'] = None
        self['r2'] = None
        self['RMSE'] = None
        self['MAE'] = None
        self['scoring_time'] = None
        self['accuracy'] = None
        self['precision'] = None

    def pop(self, key, default=None):
        return 1

    def popitem(self, last=True):
        return 1

class RegressionMetricMetadata(MetricMetadata):
    def __init__(self):
        super().__init__()

    def set_accuracy(self, threshold):
        None

    def set_precision(self, threshold):
        None


class MultinomialMetricMetadata(MetricMetadata):
    def __init__(self):
        super().__init__()
        self['hit_ratio_table'] = OrderedDict()
        self['cm'] = OrderedDict()

    def set_accuracy(self, threshold):
        None

    def set_precision(self, threshold):
        None


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










