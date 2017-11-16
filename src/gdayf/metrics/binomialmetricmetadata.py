## @package gdayf.metrics.binomialmetricmetadata
#  Define Binomial Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

from collections import OrderedDict
from gdayf.metrics.metricmetadata import MetricMetadata
from pandas import DataFrame
import numpy as np
import time
from pyspark.mllib.evaluation import MulticlassMetrics
from gdayf.conf.loadconfig import LoadLabels


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

    ## Method to set precision measure
    # Not implemented yet
    def set_precision(self, tolerance):
        pass

    ## Method to load Binomial metrics from H2OBinomialModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2OBinomialModelMetrics
    def set_h2ometrics(self, perf_metrics):
        if perf_metrics is not None:
            for parameter, _ in self.items():
                if parameter in ['gains_lift_table', 'max_criteria_and_metric_scores']:
                    try:
                        self[parameter] = perf_metrics._metric_json[parameter].as_data_frame().to_json(orient='split')
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
                                perf_metrics.confusion_matrix(
                                    metrics=each_parameter).table.as_data_frame().to_json(orient='split')
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

    ## Method to load Binomial metrics from Spark BinaryClassificationEvaluator class
    # @param self objetct pointer
    # @param evaluator BinaryClassificationEvaluator instance
    # @param objective_column  string
    # @param data as Apache Spark DataFrame
    def set_sparkmetrics(self, evaluator, data, objective_column):

        start = time.time()
        if evaluator is not None and data is not None:
                self['AUC'] = evaluator.evaluate(data,  {evaluator.metricName: "areaUnderROC"})
                self['AUPR'] = evaluator.evaluate(data, {evaluator.metricName: "areaUnderPR"})
                self['nobs'] = data.count()
                self['model_category'] = 'Binomial'
                self['max_criteria_and_metric_scores'] = None
                self['RMSE']= 10e+308

                #Generating ConfusionMatrix
                tp = data.select("prediction", data[objective_column].cast('float'))\
                    .toDF("prediction", objective_column).rdd.map(tuple)
                metrics = MulticlassMetrics(tp)
                pdf = DataFrame(data=np.array(metrics.confusionMatrix().values).reshape((2, 2)),
                                columns=['0', '1'])
                pdf['total'] = pdf.sum(axis=1)
                index = pdf.index.tolist()
                index.append('total')
                pdf = pdf.append(pdf.sum(axis=0), ignore_index=True)
                pdf.index = index
                self['cm'] = pdf.to_json(orient='split')

                self['scoring_time'] = int(time.time() - start)

