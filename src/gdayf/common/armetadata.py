from collections import OrderedDict
from json import dump

from gdayf.common.dfmetada import DFMetada
from gdayf.common.normalizationset import NormalizationSet
from gdayf.common.storagemetadata import StorageMetadata
from gdayf.metrics.metricmetadata import MetricMetadata
from gdayf.models.modelmetadata import ModelMetadata


class ArMetadata (OrderedDict):

    def __init__(self):
        super(ArMetadata, self).__init__()
        self['model_id'] = None
        self['version'] = None
        self['type'] = None
        self['objective_column'] = None
        self['timestamp'] = None
        self['load_path'] = StorageMetadata('models')
        self['metrics'] = MetricMetadata()
        self['normalizations_set'] = NormalizationSet()
        self['data_initial'] = DFMetada()
        self['data_normalized'] = DFMetada()
        self['model_parameters'] = ModelMetadata()
        self['ignored_parameters'] = None
        self['full_parameters_stack'] = None
        self['log_path'] = StorageMetadata('logs')
        self['json_path'] = StorageMetadata('json')
        self['status'] = -1

    def get_json(self):
        return dump(self, indent=4, enconding='utf8')

    def pop(self, key, default=None):
        return 1

    def popitem(self, last=True):
        return 1