from collections import OrderedDict
from gdayf.common.storagemetadata import StorageMetada
from gdayf.common.dfmetada import DFMetada
from json import dump


class ArMetadata (OrderedDict):

    def __init__(self):
        super().__init__()
        self['model_id'] = None
        self['version'] = None
        self['type'] = None
        self['objective_column'] = None
        self['timestamp'] = None
        self['load_path'] = StorageMetada('models')
        self['metrics'] = None
        self['normalizations_set'] = None
        self['data_initial'] = DFMetada()
        self['data_normalized'] = DFMetada()
        self['model_parameters'] = None
        self['ignored_parameters'] = None
        self['full_parameters_stack'] = None
        self['log_path'] = StorageMetada('logs')
        self['json_path'] = StorageMetada('json')
        self['status'] = -1

    def get_json(self):
        return dump(self, indent=4, enconding='utf8')

    def pop(self, key, default=None):
        return 1

    def popitem(self, last=True):
        return 1