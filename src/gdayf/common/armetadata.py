## @package gdayf.common.armetadata
# Define all objects, functions and structured related to Analysis_Results for one execution (final json structure)
# on OrderedDict format

from collections import OrderedDict
from json import dumps
from gdayf.common.dfmetada import DFMetada
from gdayf.common.normalizationset import NormalizationSet
from gdayf.common.storagemetadata import StorageMetadata
from gdayf.metrics.metricmetadata import MetricMetadata
from gdayf.metrics.executionmetriccollection import ExecutionMetricCollection
from gdayf.models.frameworkmetadata import FrameworkMetadata


## Class ArMetadata manage the Analysis results structs on OrderedDict format and exportable to json
class ArMetadata (OrderedDict):

    ## The constructor
    # Generate an empty AR class with all DayF objects elements initialized to correct types
    def __init__(self, type_='train'):
        OrderedDict.__init__(self)
        self['model_id'] = None
        self['version'] = None
        self['type'] = type_
        self['objective_column'] = None
        self['timestamp'] = None
        self['load_path'] = StorageMetadata().get_load_path()
        self['metrics'] = OrderedDict()
        self['normalizations_set'] = None
        self['data_initial'] = None
        self['data_normalized'] = None
        self['model_parameters'] = None
        self['ignored_parameters'] = None
        self['full_parameters_stack'] = None
        self['log_path'] = StorageMetadata().get_log_path()
        self['json_path'] = StorageMetadata().get_json_path()
        self['status'] = -1

    ## Get json format string associted to class OredredDict parameters with encoding utf-8 and indent = 4
    # @param self object pointer
    # @return OrderedDict() structure associated to json file with indent 4 and encoding utf-8
    def get_json(self):
        return dumps(self, indent=4)

    ## Anulate pop fetures from OrderedDict parent class.
    # Stability  proposals
    def pop(self, key, default=None):
        return 1

    ## Anulate popitem fetures from OrderedDict parent class
    # Stability  proposals
    def popitem(self, last=True):
        return 1
## Main block only for testing issues
if __name__ == "__main__":
    ## Varible for testinf propouses
    m = ArMetadata()
    print(m.get_json())

