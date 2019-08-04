## @package gdayf.metrics.metricmetadata
#  Define Base Metric object as OrderedDict() of common measures for all metrics types
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













