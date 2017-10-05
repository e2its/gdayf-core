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
        self['cm']['min_per_class_accuracy'] = None
        self['cm']['absolute_mcc'] = None
        self['cm']['precision'] = None
        self['cm']['accuracy'] = None
        self['cm']['f0point5'] = None
        self['cm']['f2'] = None
        self['cm']['f1'] = None

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
                try:
                    self[parameter] = perf_metrics._metric_json[parameter].as_data_frame().to_json(orient='split')
                except KeyError as kexecution_error:
                    print(repr(kexecution_error))
                except AttributeError as aexecution_error:
                    print(repr(aexecution_error))
                except TypeError as texecution_error:
                    print(repr(texecution_error))
            elif parameter in ['cm']:
                for each_parameter, __ in self['cm'].items():
                    try:
                        self['cm'][each_parameter] = \
                            perf_metrics.confusion_matrix(
                                metrics=each_parameter).table.as_data_frame().to_json(orient='split')
                    except KeyError as kexecution_error:
                        print(repr(kexecution_error))
                    except AttributeError as aexecution_error:
                        print(repr(aexecution_error))
                    except TypeError as texecution_error:
                        print(repr(texecution_error))
            elif parameter in ['thresholds_and_metric_scores']:
                pass
            else:
                try:
                    self[parameter] = perf_metrics._metric_json[parameter]
                except KeyError as kexecution_error:
                    print(repr(kexecution_error))
                except AttributeError as aexecution_error:
                    print(repr(aexecution_error))
