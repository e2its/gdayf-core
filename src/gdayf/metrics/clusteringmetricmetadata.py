## @package gdayf.metrics.clusteringmetricmetadata
# Define Clustering Metric object as OrderedDict() of common measures for all frameworks
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

'''
The basic ones
$betweenss: is the between clusters sum of squares. In fact it is the mean of distances between cluster centers.
One expects, this ratio, to be as higher as possible, since we would like to have heterogenous clusters.
2 · ( ∑m ∑n | CmP - CnP |2 )  /  p · p - 1

$withinss: is the within cluster sum of squares. So it results in a vector with a number for each cluster.
One expects, this ratio, to be as lower as possible for each cluster,
since we would like to have homogeneity within the clusters.
( ∑m  | Xm - C |2 )  /  p

Some equalities may help to understand:
$tot.withinss = sum ( $withinss )
$totss = $tot.withinss + $betweenss
'''
from gdayf.metrics.metricmetadata import MetricMetadata
from pandas import DataFrame
from collections import OrderedDict
import json
import time


# Class Base for Regression metricts as OrderedDict
#
# Base Metrics for Clustering
# [No expanded metrics]
class ClusteringMetricMetadata(MetricMetadata):

    ## Method constructor
    # @param self object pointer
    def __init__(self):
        MetricMetadata.__init__(self)
        self['betweenss'] = None
        self['tot_withinss'] = None
        self['totss'] = None
        self['centroid_stats'] = None

    ## Method to set precision measure
    # Not implemented yet
    def set_precision(self, threshold):
        pass

    ## Method to load Clustering metrics from H2OClusteringModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2ORegressionModelMetrics
    def set_h2ometrics(self, perf_metrics):
        for parameter, _ in self.items():
            try:
                if perf_metrics is not None:
                    if parameter == 'centroid_stats':
                        self[parameter] = perf_metrics._metric_json[parameter].as_data_frame()
                        self['k'] = int(self[parameter]['centroid'].max())
                        self[parameter] = json.loads(self[parameter].to_json(orient='split'),
                                                     object_pairs_hook=OrderedDict)
                    else:
                        self[parameter] = perf_metrics._metric_json[parameter]
            except KeyError as kexecution_error:
                pass
                #print('Trace: ' + repr(kexecution_error))
            except AttributeError as aexecution_error:
                print('Trace: ' + repr(aexecution_error))
            except TypeError as texecution_error:
                print('Trace: ' + repr(texecution_error))

    ## Method to load Regression metrics from Spark RegressionEvaluator class
    # @param self objetct pointer
    # @param model BisectingKMeans|KMeans instance
    # @param columns  columns applied to build Cluster
    # @param data as Apache Spark DataFrame
    def set_sparkmetrics(self, model, data):

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
