## @package gdayf.handler_metrics.h2oclusteringmetricmetadata
# Define Clustering Metric object as OrderedDict() of common measures for all frameworks
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
from gdayf.metrics.clusteringmetricmetadata import ClusteringMetricMetadata
from collections import OrderedDict
import json


# Class Base for Regression metricts as OrderedDict
#
# Base Metrics for Clustering
# [No expanded metrics]
class H2OClusteringMetricMetadata(ClusteringMetricMetadata):

    ## Method constructor
    # @param self object pointer
    def __init__(self):
        ClusteringMetricMetadata.__init__(self)


    ## Method to load Clustering metrics from H2OClusteringModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2ORegressionModelMetrics
    def set_metrics(self, perf_metrics):
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

