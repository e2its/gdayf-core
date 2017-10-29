## @package gdayf.metrics.regressionmetricmetadata
# Define Regression Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

from gdayf.metrics.metricmetadata import MetricMetadata
import time


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

    ## Method to load Regression metrics from Spark RegressionEvaluator class
    # @param self objetct pointer
    # @param evaluator RegressionEvaluator instance
    # @param objective_column  string
    # @param data as Apache Spark DataFrame
    def set_sparkmetrics(self, evaluator, data, objective_column=None):

        start = time.time()
        if evaluator is not None and data is not None:
                self['MSE'] = evaluator.evaluate(data, {evaluator.metricName: "mse"})
                self['mean_residual_deviance'] = None
                self['nobs'] = data.count()
                self['model_category'] = 'Regression'
                self['predictions'] = None
                self['rmsle'] = None
                self['r2'] = evaluator.evaluate(data, {evaluator.metricName: "r2"})
                self['RMSE'] = evaluator.evaluate(data, {evaluator.metricName: "rmse"})
                self['MAE'] = evaluator.evaluate(data, {evaluator.metricName: "mae"})
                self['scoring_time'] = int(time.time() - start)

