## @package gdayf.handler_metrics.h2omultinomialmetricmetadata
# Define Multinomial Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
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
class H2OMultinomialMetricMetadata(MultinomialMetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        MultinomialMetricMetadata.__init__(self)

    ## Method to load MultiNomial metrics from H2OMultinomialModelMetrics class
    # @param self objetct pointer
    # @param perf_metrics H2OMultinomialModelMetrics
    def set_metrics(self, perf_metrics):
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

