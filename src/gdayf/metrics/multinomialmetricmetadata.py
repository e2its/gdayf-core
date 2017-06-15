from collections import OrderedDict
from gdayf.metrics.metricmetadata import MetricMetadata
##  Define Multinomial Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way


# Class Base for Multinomial metricts as OrderedDict
#
# Base Metrics for Multinomial
# [hit_ratio_table (as json_dataframe(orient=split)), cm (as json_dataframe(orient=split) structure)]
class MultinomialMetricMetadata(MetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        MetricMetadata.__init__(self)
        self['hit_ratio_table'] = OrderedDict()
        self['cm'] = OrderedDict()

    # Me#thod to set precision measure
    # Not implemented yet
    def set_precision(self, threshold):
        pass

    ## Method to load Binomial metrics from H2OMultinomialModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2OMultinomialModelMetrics
    def set_h2ometrics(self, perf_metrics):
        for parameter, _ in self.items():
            if parameter in ['hit_ratio_table']:
                self[parameter] = perf_metrics._metric_json[parameter].as_data_frame().to_json(orient='split')
            elif parameter in ['cm']:
                self[parameter] = \
                    perf_metrics._metric_json[parameter]['table'].as_data_frame().to_json(orient='split')
            else:
                try:
                    self[parameter] = perf_metrics._metric_json[parameter]
                except KeyError:
                    pass