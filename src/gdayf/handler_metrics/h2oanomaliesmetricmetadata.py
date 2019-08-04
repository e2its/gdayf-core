## @package gdayf.handler_metrics.h2oanomaliesmetricmetadata
# Define Regression Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

'''
 * This file is part of the gDayF AutoML Core Framework project
 * distribution (https://github.com/e2its/gdayf-core).
 * Copyright (c) 2016-2019 Jose Luis Sanchez del Coso <e2its.es@gmail.com>.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ** Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

from gdayf.handler_metrics.h2oregressionmetricmetadata import H2ORegressionMetricMetadata


## Class Base for Regression metricts as OrderedDict
# Base Metrics for Regression
# [No expanded metrics]
class H2OAnomaliesMetricMetadata(H2ORegressionMetricMetadata):

    ## Method constructor
    # @param self object pointer
    def __init__(self):
        H2ORegressionMetricMetadata.__init__(self)



