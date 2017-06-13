from collections import OrderedDict


class MetricMetadata(OrderedDict):
    def __init__(self):
        super(MetricMetadata, self).__init__()
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


    def pop(self, key, default=None):
        return 1

    def popitem(self, last=True):
        return 1










