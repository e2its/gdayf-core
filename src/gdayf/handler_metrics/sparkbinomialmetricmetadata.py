## @package gdayf.handler_metrics.sparkbinomialmetricmetadata
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
from pandas import DataFrame
import numpy as np
import time
from pyspark.mllib.evaluation import MulticlassMetrics
import json


##Class Base for Binomial metricts as OrderedDict
#
# Base Metrics for binomial
# [AUC, gains_lift_table (as json_dataframe(orient=split)), Gini,
# mean_per_class_error, logloss, max_criteria_and_metric_scores (as json_dataframe(orient=split)), cm (next)]
# cm[min_per_class_accuracy, absolute_mcc, precision, accuracy, f0point5, f2, f1]
# as json_dataframe(orient=split) structure
class SparkBinomialMetricMetadata(BinomialMetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        BinomialMetricMetadata.__init__(self)

    ## Method to load Binomial metrics from Spark BinaryClassificationEvaluator class
    # @param self objetct pointer
    # @param evaluator BinaryClassificationEvaluator instance
    # @param objective_column  string
    # @param data as Apache Spark DataFrame
    def set_metrics(self, evaluator, data, objective_column):

        start = time.time()
        if evaluator is not None and data is not None:
                self['AUC'] = evaluator.evaluate(data,  {evaluator.metricName: "areaUnderROC"})
                self['AUPR'] = evaluator.evaluate(data, {evaluator.metricName: "areaUnderPR"})
                self['nobs'] = data.count()
                self['model_category'] = 'Binomial'
                self['max_criteria_and_metric_scores'] = None
                self['RMSE'] = 10e+308

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
                self['cm'] = json.loads(pdf.to_json(orient='split'), object_pairs_hook=OrderedDict)

                self['scoring_time'] = int(time.time() - start)


