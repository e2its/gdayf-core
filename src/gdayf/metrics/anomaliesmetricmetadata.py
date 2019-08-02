## @package gdayf.metrics.anomaliesmetricmetadata
# Define Regression Anomalies Metric object as OrderedDict() of common measures for all frameworks
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

from gdayf.metrics.regressionmetricmetadata import RegressionMetricMetadata

# Class Base for Regression metrics as OrderedDict
#
# Base Metrics for Anomalies-Regression
# [No expanded metrics]
class AnomaliesMetricMetadata(RegressionMetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        RegressionMetricMetadata.__init__(self)


