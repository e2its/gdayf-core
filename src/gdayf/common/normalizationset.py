## @package gdayf.common.normalizationset

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

from collections import OrderedDict


## Class NormalizationSet manage the Normalizations metadata as OrderedDict supporting Normalizer Class methods and actions
class NormalizationSet (OrderedDict):

    ## The constructor
    # Generate an empty NormalizationSet class with all elements initialized to correct types
    def __init__(self):
        OrderedDict.__init__(self)
        self.reset()

    ## Method oriented to reset a NormalizationSet instance
    # @param self  object pointer
    def reset(self):
        self['type'] = None
        self['class'] = None
        self['objective'] = OrderedDict()

    ## Method oriented to establish base Normalization [Metadata]
    # @param self  object pointer
    def set_base(self, datetime=True):
        self.reset()
        self['type'] = "normalization"
        self['class'] = "base"
        self['datetime'] = datetime
        self['objective']['value'] = OrderedDict()

    ## Method oriented to establish ignore_column Normalization [Metadata]
    # @param self  object pointer
    def set_ignore_column(self):
        self.reset()
        self['type'] = "drop"
        self['class'] = "ignore_column"
        self['objective']['value'] = OrderedDict()

    ## Method oriented to establish stdmean Normalization [Metadata]
    # @param self  object pointer
    # @param mean average data value
    # @param std standard deviation data value
    def set_stdmean(self, mean=0, std=1):
        self.reset()
        self['type'] = "normalization"
        self['class'] = "stdmean"
        self['objective']['value'] = OrderedDict()
        self['objective']['mean'] = mean
        self['objective']['std'] = std

    ## Method oriented to establish drop_missing Normalization [Metadata]
    # @param self  object pointer
    def set_drop_missing(self):
        self.reset()
        self['type'] = "drop"
        self['class'] = "drop_missing"
        self['objective']['value'] = OrderedDict()

    ## Method oriented to establish bucketing actions [Metadata]
    # @param self  object pointer
    # @param buckets_number Number of buckets to be implemented
    # @param fixed_size True for interval fixed size, False for working on Frequency basis
    def set_discretize(self, buckets_number=10, fixed_size=True):
        self.reset()
        self['type'] = "bucketing"
        self['class'] = "discretize"
        self['objective']['buckets_number'] = buckets_number
        self['objective']['fixed_size'] = fixed_size

    ## Method oriented to establish re-scaling data actions [Metadata]
    # @param self  object pointer
    # @param minval Minimal value on source Dataframe
    # @param maxval Maximum value on source Dataframe
    # @param minrange Minimal value on target Dataframe
    # @param minval Maximum value on target Dataframe
    def set_working_range(self, minval=-1.0, maxval=1.0, minrange = -1.0, maxrange = 1.0):
        self.reset()
        self['type'] = "normalization"
        self['class'] = "working_range"
        self['objective']['minval'] = minval
        self['objective']['maxval'] = maxval
        self['objective']['minval'] = minrange
        self['objective']['maxval'] = maxrange

    ## Method oriented to establish offset (+ or -) data actions [Metadata]
    # @param self  object pointer
    # @param offset offset value to be applied into source Dataframe
    def set_offset(self, offset=0):
        self.reset()
        self['type'] = "normalization"
        self['class'] = "offset"
        self['objective']['offset'] = offset

    ## Method oriented to aggregate minimal and maximal non-frequent values on aggregated intervals [Metadata]
    # Not implemented on Normalizer class [Not Use]
    # @param self  object pointer
    # @param bucket_ratio ratio for aggregation ratio based on distribution (0.25 means 12.5% minimal and 12.5% maximal)
    def set_aggregation(self, bucket_ratio=0.25):
        self.reset()
        self['type'] = "bucketing"
        self['class'] = "working_range"
        self['objective']['bucket_ratio'] = bucket_ratio

    ## Method oriented to establish fixed value imputation data actions into missing values [Metadata]
    # @param self  object pointer
    # @param value value to be applied into source Dataframe missing values
    def set_fixed_missing_values(self, value=0.0):
        self.reset()
        self['type'] = "imputation"
        self['class'] = "fixed_missing_values"
        self['objective']['value'] = value

    ## Method oriented to establish variable value imputation data actions into missing values based on objective column [Metadata]
    # @param self  object pointer
    # @param objective_column Objective column string type identification
    # @param full True establishes all value as mean ignoring objective_column. False establishes mean of coincident
    #  values on objective column
    def set_mean_missing_values(self, objective_column, full=False):
        self.reset()
        self['type'] = "imputation"
        self['class'] = "mean_missing_values"
        self['objective']['objective_column'] = objective_column
        self['objective']['full'] = full

    ## Method oriented to establish variable extrapolated value imputation data actions into missing values based
    # on objective column [Metadata]
    # @param self  object pointer
    # @param objective_column Objective column string type identification
    def set_progressive_missing_values(self, objective_column):
        self.reset()
        self['type'] = "imputation"
        self['class'] = "progressive_missing_values"
        self['objective']['objective_column'] = objective_column

