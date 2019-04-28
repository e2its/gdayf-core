## @package gdayf.handler_metrics.h2oregressionmetricmetadata
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

from gdayf.metrics.regressionmetricmetadata import RegressionMetricMetadata
import time
from numpy import isnan


## Class Base for Regression metricts as OrderedDict
# Base Metrics for Regression
# [No expanded metrics]
class H2ORegressionMetricMetadata(RegressionMetricMetadata):

    ## Method constructor
    # @param self object pointer
    def __init__(self):
        RegressionMetricMetadata.__init__(self)

    ## Method to load Regression metrics from H2ORegressionModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2ORegressionModelMetrics
    def set_metrics(self, perf_metrics):
        if perf_metrics is not None:
            for parameter, _ in self.items():
                try:
                    self[parameter] = perf_metrics._metric_json[parameter]
                except KeyError as kexecution_error:
                    #print('Trace: ' + repr(kexecution_error))
                    pass
                except AttributeError as aexecution_error:
                    #print('Trace: ' + repr(aexecution_error))
                    pass


