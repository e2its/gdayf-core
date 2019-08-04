## @package gdayf.handler_metrics.sparkclusteringmetricmetadata
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

from gdayf.metrics.clusteringmetricmetadata import ClusteringMetricMetadata
from pandas import DataFrame
from collections import OrderedDict
import json
import time


## Class Base for Clustering metricts as OrderedDict
# Base Metrics for Regression
# [No expanded metrics]
class SparkClusteringMetricMetadata(ClusteringMetricMetadata):

    ## Method constructor
    # @param self object pointer
    def __init__(self):
        ClusteringMetricMetadata.__init__(self)

    ## Method to load Regression metrics from Spark RegressionEvaluator class
    # @param self objetct pointer
    # @param model BisectingKMeans|KMeans instance
    # @param columns  columns applied to build Cluster
    # @param data as Apache Spark DataFrame
    def set_metrics(self, model, data):

        start = time.time()
        if model is not None and data is not None:
            self['clusterCenters'] = DataFrame(model.clusterCenters())
            self['k'] = self['clusterCenters'].shape[0]
            self['clusterCenters'] = json.loads(DataFrame(model.clusterCenters()).to_json(orient='split'),
                                                object_pairs_hook=OrderedDict)
            self['bettweenss'] = 1e15  # Need to be implemented
            self['totss'] = 1e15  # Need to be implemented
            self['tot_withinss'] = model.computeCost(data)
            self['nobs'] = data.count()
            self['model_category'] = 'Clustering'
            self['predictions'] = None
            self['RMSE'] = 10e+308
            self['scoring_time'] = int(time.time() - start)

