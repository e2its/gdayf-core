from collections import OrderedDict

class NormalizationSet (OrderedDict):
    def __init__(self):
        OrderedDict.__init__(self)
        self.reset()

    def reset(self):
        self['type'] = None
        self['class'] = None
        self['objective'] = OrderedDict()

    def set_base(self):
        self.reset()
        self['type'] = "normalization"
        self['class'] = "base"
        self['objective']['value'] = OrderedDict()

    def set_ignore_column(self):
        self.reset()
        self['type'] = "drop"
        self['class'] = "ignore_column"
        self['objective']['value'] = OrderedDict()

    def set_stdmean(self):
        self.reset()
        self['type'] = "normalization"
        self['class'] = "stdmean"
        self['objective']['value'] = OrderedDict()

    def set_drop_missing(self):
        self.reset()
        self['type'] = "drop"
        self['class'] = "drop_missing"
        self['objective']['value'] = OrderedDict()

    def set_discretize(self, buckets_number=10, fixed_size=True):
        self.reset()
        self['type'] = "bucketing"
        self['class'] = "discretize"
        self['objective']['buckets_number'] = buckets_number
        self['objective']['fixed_size'] = fixed_size

    def set_working_range(self, minval=-1.0, maxval=1.0):
        self.reset()
        self['type'] = "normalization"
        self['class'] = "working_range"
        self['objective']['minval'] = minval
        self['objective']['maxval'] = maxval

    def set_aggregation(self, bucket_ratio=0.25):
        self.reset()
        self['type'] = "bucketing"
        self['class'] = "working_range"
        self['objective']['bucket_ratio'] = bucket_ratio

    def set_fixed_missing_values(self, value=0.0):
        self.reset()
        self['type'] = "imputation"
        self['class'] = "fixed_missing_values"
        self['objective']['value'] = value

    def set_mean_missing_values(self, objective_column, full=False):
        self.reset()
        self['type'] = "imputation"
        self['class'] = "mean_missing_values"
        self['objective']['objective_column'] = objective_column
        self['objective']['full'] = full

    def set_progressive_missing_values(self, objective_column):
        self.reset()
        self['type'] = "imputation"
        self['class'] = "progressive_missing_values"
        self['objective']['objective_column'] = objective_column

