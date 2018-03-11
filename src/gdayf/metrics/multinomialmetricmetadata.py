## @package gdayf.metrics.multinomialmetricmetadata
# Define Multinomial Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

from collections import OrderedDict
from gdayf.metrics.metricmetadata import MetricMetadata
from pandas import DataFrame
import numpy as np
import time
import json
from pyspark.mllib.evaluation import MulticlassMetrics


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

    ## Method to set precision measure
    # Not implemented yet
    def set_precision(self, threshold):
        pass

    ## Method to load Binomial metrics from H2OMultinomialModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2OMultinomialModelMetrics
    def set_h2ometrics(self, perf_metrics):
        if perf_metrics is not None:
            for parameter, _ in self.items():
                if parameter in ['hit_ratio_table']:
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
                    try:
                        self[parameter] = \
                            json.loads(
                                perf_metrics._metric_json[parameter]['table'].as_data_frame().to_json(orient='split'),
                                object_pairs_hook=OrderedDict)
                    except KeyError as kexecution_error:
                        pass
                        #print('Trace: ' + repr(kexecution_error))
                    except AttributeError as aexecution_error:
                        print('Trace: ' + repr(aexecution_error))
                    except TypeError as texecution_error:
                        print('Trace: ' + repr(texecution_error))
                else:
                    try:
                        self[parameter] = perf_metrics._metric_json[parameter]
                    except KeyError as kexecution_error:
                        pass
                        #print('Trace: ' + repr(kexecution_error))
                    except AttributeError as aexecution_error:
                        print('Trace: ' + repr(aexecution_error))

    ## Method to load Multinomial metrics from Spark MulticlassClassificationEvaluator class
    # @param self objetct pointer
    # @param evaluator MulticlassClassificationEvaluator instance
    # @param objective_column  string
    # @param data as Apache Spark DataFrame
    def set_sparkmetrics(self, evaluator, data, objective_column):

        start = time.time()
        if evaluator is not None and data is not None:
                self['f1'] = evaluator.evaluate(data,  {evaluator.metricName: "f1"})
                self['weightedPrecision'] = evaluator.evaluate(data, {evaluator.metricName: "weightedPrecision"})
                self['weightedRecall'] = evaluator.evaluate(data, {evaluator.metricName: "weightedRecall"})
                self['accuracy'] = evaluator.evaluate(data, {evaluator.metricName: "accuracy"})
                self['nobs'] = data.count()
                self['model_category'] = 'Multinomial'
                self['RMSE'] = 10e+308

                #Generating ConfusionMatrix
                dimcount = data.select(objective_column).distinct().count()
                tp = data.select("prediction", data[objective_column].cast('float'))\
                    .toDF("prediction", objective_column).rdd.map(tuple)
                metrics = MulticlassMetrics(tp)
                pdf = DataFrame(data=np.array(metrics.confusionMatrix().values).reshape((dimcount, dimcount)),
                                columns=range(0, dimcount))
                pdf['total'] = pdf.sum(axis=1)
                index = pdf.index.tolist()
                index.append('total')
                pdf = pdf.append(pdf.sum(axis=0), ignore_index=True)
                pdf.index = index
                self['cm'] = json.loads(pdf.to_json(orient='split'), object_pairs_hook=OrderedDict)

                self['scoring_time'] = int(time.time() - start)
