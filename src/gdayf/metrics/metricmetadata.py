## @package gdayf.metrics.metricmetadata
#  Define Base Metric object as OrderedDict() of common measures for all metrics types
#  on an unified way

from collections import OrderedDict

## Class Base for metricts as OrderedDict
#
# Base Metrics
# [MSE, mean_residual_deviance, nobs, predictions, rmsle, r2, RMSE,  MAE, scoring_time]
class MetricMetadata(OrderedDict):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        OrderedDict.__init__(self)
        self['MSE'] = None
        self['mean_residual_deviance'] = None
        self['nobs'] = None
        self['model_category'] = None
        self['predictions'] = None
        self['rmsle'] = None
        self['r2'] = None
        self['RMSE'] = None
        self['MAE'] = None
        self['scoring_time'] = None

    ## Method pop disable OrderedDict().pop method
    # compatibility with OrderdDict().pop
    def pop(self, key, default=None):
        return 1
    ## Method constructor
    # compatibility with OrderdDict().popitem
    def popitem(self, last=True):
        return 1










