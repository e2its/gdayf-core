## @package gdayf.metrics.binomialmetricmetadata
#  Define Binomial Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

from collections import OrderedDict
from gdayf.metrics.metricmetadata import MetricMetadata


##Class Base for Binomial metricts as OrderedDict
#
# Base Metrics for binomial
# [AUC, gains_lift_table (as json_dataframe(orient=split)), Gini,
# mean_per_class_error, logloss, max_criteria_and_metric_scores (as json_dataframe(orient=split)), cm (next)]
# cm[min_per_class_accuracy, absolute_mcc, precision, accuracy, f0point5, f2, f1]
# as json_dataframe(orient=split) structure
class BinomialMetricMetadata(MetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        MetricMetadata.__init__(self)
        self['AUC'] = None
        self['gains_lift_table'] = OrderedDict()
        self['Gini'] = None
        self['mean_per_class_error'] = None
        self['logloss'] = None
        self['max_criteria_and_metric_scores'] = OrderedDict
        self['cm'] = OrderedDict()
        self['cm']['min_per_class_accuracy'] = OrderedDict()
        self['cm']['absolute_mcc'] = OrderedDict()
        self['cm']['precision'] = OrderedDict()
        self['cm']['accuracy'] = OrderedDict()
        self['cm']['f0point5'] = OrderedDict()
        self['cm']['f2'] = OrderedDict()
        self['cm']['f1'] = OrderedDict()

    # Me#thod to set precision measure
    # Not implemented yet
    def set_precision(self, tolerance):
        pass

    ## Method to load Binomial metrics from H2OBinomialModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2OBinomialModelMetrics
    def set_h2ometrics(self, perf_metrics):
        for parameter, _ in self.items():
            if parameter in ['gains_lift_table', 'max_criteria_and_metric_scores']:
                self[parameter] = perf_metrics._metric_json[parameter].as_data_frame().to_json(orient='split')
            elif parameter in ['cm']:
                pass
            elif parameter in ['thresholds_and_metric_scores']:
                self['cm'] = OrderedDict()
                for each_parameter in ['min_per_class_accuracy', 'absolute_mcc', 'precision', 'accuracy',
                                       'f0point5', 'f2', 'f1', 'mean_per_class_accuracy']:
                    self['cm'][each_parameter] = \
                        perf_metrics.confusion_matrix(
                            metrics=each_parameter).table.as_data_frame().to_json(orient='split')
            else:
                try:
                    self[parameter] = perf_metrics._metric_json[parameter]
                except KeyError:
                    pass
