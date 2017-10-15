## @package gdayf.metrics.clusteringmetricmetadata
# Define Clustering Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

from gdayf.metrics.metricmetadata import MetricMetadata


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
                        self[parameter] = self[parameter].to_json(orient='split')
                    else:
                        self[parameter] = perf_metrics._metric_json[parameter]
            except KeyError as kexecution_error:
                pass
                #print('Trace: ' + repr(kexecution_error))
            except AttributeError as aexecution_error:
                print('Trace: ' + repr(aexecution_error))
            except TypeError as texecution_error:
                print('Trace: ' + repr(texecution_error))