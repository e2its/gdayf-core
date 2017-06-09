from collections import OrderedDict

from gdayf.metrics.metricmetadata import MetricMetadata


class MultinomialMetricMetadata(MetricMetadata):
    def __init__(self):
        super(MultinomialMetricMetadata, self).__init__()
        self['hit_ratio_table'] = OrderedDict()
        self['cm'] = OrderedDict()

    def set_accuracy(self, threshold):
        None

    def set_precision(self, threshold):
        None

    def set_metrics(self, perf_metrics):
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
                    None