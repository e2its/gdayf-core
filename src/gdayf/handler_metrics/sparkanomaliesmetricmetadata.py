## @package gdayf.handler_metrics.sparkanomaliesmetricmetadata
# Define Regression Metric object as OrderedDict() of common measures for all frameworks
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

from gdayf.metrics.anomaliesmetricmetadata import AnomaliesMetricMetadata
import time
from numpy import isnan


## Class Base for Regression metricts as OrderedDict
# Base Metrics for Regression
# [No expanded metrics]
class SparkAnomaliesMetricMetadata(AnomaliesMetricMetadata):

    ## Method constructor
    # @param self object pointer
    def __init__(self):
        AnomaliesMetricMetadata.__init__(self)

