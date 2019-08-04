## @package gdayf.handler_metrics.sparkregressionmetricmetadata
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

from gdayf.metrics.regressionmetricmetadata import RegressionMetricMetadata
import time
from numpy import isnan


## Class Base for Regression metricts as OrderedDict
# Base Metrics for Regression
# [No expanded metrics]
class SparkRegressionMetricMetadata(RegressionMetricMetadata):

    ## Method constructor
    # @param self object pointer
    def __init__(self):
        RegressionMetricMetadata.__init__(self)

    ## Method to load Regression metrics from Spark RegressionEvaluator class
    # @param self objetct pointer
    # @param evaluator RegressionEvaluator instance
    # @param objective_column  string
    # @param data as Apache Spark DataFrame
    def set_metrics(self, evaluator, data, objective_column=None):

        start = time.time()
        if evaluator is not None and data is not None:
                self['MSE'] = evaluator.evaluate(data, {evaluator.metricName: "mse"})
                self['mean_residual_deviance'] = None
                self['nobs'] = data.count()
                self['model_category'] = 'Regression'
                self['predictions'] = None
                self['rmsle'] = None
                self['r2'] = evaluator.evaluate(data, {evaluator.metricName: "r2"})
                self['RMSE'] = evaluator.evaluate(data, {evaluator.metricName: "rmse"})
                self['MAE'] = evaluator.evaluate(data, {evaluator.metricName: "mae"})
                self['scoring_time'] = int(time.time() - start)
                if isnan(self['RMSE']):
                    self['RMSE'] = 1e+16
                if isnan(self['r2']):
                    self['r2'] = 0


