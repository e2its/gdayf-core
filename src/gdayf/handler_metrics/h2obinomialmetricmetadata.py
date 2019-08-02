## @package gdayf.handler_metrics.h2obinomialmetricmetadata
#  Define Binomial Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

from collections import OrderedDict
from gdayf.metrics.binomialmetricmetadata import BinomialMetricMetadata
import json


##Class Base for Binomial metricts as OrderedDict
#
# Base Metrics for binomial
# [AUC, gains_lift_table (as json_dataframe(orient=split)), Gini,
# mean_per_class_error, logloss, max_criteria_and_metric_scores (as json_dataframe(orient=split)), cm (next)]
# cm[min_per_class_accuracy, absolute_mcc, precision, accuracy, f0point5, f2, f1]
# as json_dataframe(orient=split) structure
class H2OBinomialMetricMetadata(BinomialMetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        BinomialMetricMetadata.__init__(self)

    ## Method to set precision measure
    # Not implemented yet
    def set_precision(self, tolerance):
        pass

    ## Method to load Binomial metrics from H2OBinomialModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2OBinomialModelMetrics
    def set_metrics(self, perf_metrics):
        if perf_metrics is not None:
            for parameter, _ in self.items():
                if parameter in ['gains_lift_table', 'max_criteria_and_metric_scores']:
                    try:
                        self[parameter] = json.loads(
                            perf_metrics._metric_json[parameter].as_data_frame().to_json(orient='split'),
                            object_pairs_hook=OrderedDict)
                    except KeyError as kexecution_error:
                        pass
                        #print('Trace: ' + repr(kexecution_error))
                    except AttributeError as aexecution_error:
                        print('Trace: ' + repr(aexecution_error))
                    except TypeError as texecution_error:
                        print('Trace: ' + repr(texecution_error))
                elif parameter in ['cm']:
                    for each_parameter, __ in self['cm'].items():
                        try:
                            self['cm'][each_parameter] = \
                                json.loads(
                                    perf_metrics.confusion_matrix(
                                        metrics=each_parameter).table.as_data_frame().to_json(orient='split'),
                                    object_pairs_hook=OrderedDict)
                        except KeyError as kexecution_error:
                            pass
                            #print('Trace: ' + repr(kexecution_error))
                        except AttributeError as aexecution_error:
                            print('Trace: ' + repr(aexecution_error))
                        except TypeError as texecution_error:
                            print('Trace: ' + repr(texecution_error))
                        except ValueError as vexecution_error:
                            print(repr(vexecution_error))
                elif parameter in ['thresholds_and_metric_scores']:
                    pass
                else:
                    try:
                        self[parameter] = perf_metrics._metric_json[parameter]
                    except KeyError as kexecution_error:
                        pass
                        #print('Trace: ' + repr(kexecution_error))
                    except AttributeError as aexecution_error:
                        print('Trace: ' + repr(aexecution_error))



