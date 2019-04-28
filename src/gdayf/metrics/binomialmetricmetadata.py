## @package gdayf.metrics.binomialmetricmetadata
#  Define Binomial Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
'''

from collections import OrderedDict
from gdayf.metrics.metricmetadata import MetricMetadata
from pandas import DataFrame
import numpy as np
import time
from pyspark.mllib.evaluation import MulticlassMetrics
import json


##Class Base for Binomial metricts as OrderedDict
#
# Base Metrics for binomial
# [AUC, gains_lift_table (as json_dataframe(orient=split)), Gini,
# mean_per_class_error, logloss, max_criteria_and_metric_scores (as json_dataframe(orient=split)), cm (next)]
# cm[min_per_class_accuracy, absolute_mcc, precision, accuracy, f0point5, f2, f1]
# as json_dataframe(orient=split) structure
class BinomialMetricMetadata(MetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        MetricMetadata.__init__(self)
        self['AUC'] = None
        self['gains_lift_table'] = OrderedDict()
        self['Gini'] = None
        self['mean_per_class_error'] = None
        self['logloss'] = None
        self['max_criteria_and_metric_scores'] = OrderedDict
        self['cm'] = OrderedDict()
        self['cm']['min_per_class_accuracy'] = None
        self['cm']['absolute_mcc'] = None
        self['cm']['precision'] = None
        self['cm']['accuracy'] = None
        self['cm']['f0point5'] = None
        self['cm']['f2'] = None
        self['cm']['f1'] = None

    ## Method to set precision measure
    # Not implemented yet
    def set_precision(self, tolerance):
        pass

