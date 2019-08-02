## @package gdayf.metrics.multinomialmetricmetadata
# Define Multinomial Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

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
from gdayf.metrics.metricmetadata import MetricMetadata
from pandas import DataFrame
import numpy as np
import time
import json
from pyspark.mllib.evaluation import MulticlassMetrics


# Class Base for Multinomial metricts as OrderedDict
#
# Base Metrics for Multinomial
# [hit_ratio_table (as json_dataframe(orient=split)), cm (as json_dataframe(orient=split) structure)]
class MultinomialMetricMetadata(MetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        MetricMetadata.__init__(self)
        self['hit_ratio_table'] = OrderedDict()
        self['cm'] = OrderedDict()

    ## Method to set precision measure
    # Not implemented yet
    def set_precision(self, threshold):
        pass

