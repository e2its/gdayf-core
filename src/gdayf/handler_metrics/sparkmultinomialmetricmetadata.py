## @package gdayf.handler_metrics.sparkmultinomialmetricmetadata
# Define Multinomial Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

'''
 * This file is part of the gDayF AutoML Core Framework project
 * distribution (https://github.com/e2its/gdayf-core).
 * Copyright (c) 2016-2019 Jose Luis Sanchez del Coso <e2its.es@gmail.com>.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ** Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

from collections import OrderedDict
from gdayf.metrics.multinomialmetricmetadata import MultinomialMetricMetadata
from pandas import DataFrame
import numpy as np
import time
import json
from pyspark.mllib.evaluation import MulticlassMetrics


# Class Base for Multinomial metricts as OrderedDict
#
# Base Metrics for Multinomial
# [hit_ratio_table (as json_dataframe(orient=split)), cm (as json_dataframe(orient=split) structure)]
class SparkMultinomialMetricMetadata(MultinomialMetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        MultinomialMetricMetadata.__init__(self)

    ## Method to load Multinomial metrics from Spark MulticlassClassificationEvaluator class
    # @param self objetct pointer
    # @param evaluator MulticlassClassificationEvaluator instance
    # @param objective_column  string
    # @param data as Apache Spark DataFrame
    def set_metrics(self, evaluator, data, objective_column):

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
