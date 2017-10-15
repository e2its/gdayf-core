## @package gdayf.metrics.regressionmetricmetadata
# Define Regression Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

from gdayf.metrics.metricmetadata import MetricMetadata


## Class Base for Regression metricts as OrderedDict
# Base Metrics for Regression
# [No expanded metrics]
class RegressionMetricMetadata(MetricMetadata):

    ## Method constructor
    # @param self object pointer
    def __init__(self):
        MetricMetadata.__init__(self)

    ## Method to set precision measure
    # Not implemented yet
    def set_precision(self, threshold):
        pass

    ## Method to load Regression metrics from H2ORegressionModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2ORegressionModelMetrics
    def set_h2ometrics(self, perf_metrics):
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


