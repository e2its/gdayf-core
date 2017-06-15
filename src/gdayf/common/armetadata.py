from collections import OrderedDict
from json import dump
from gdayf.common.dfmetada import DFMetada
from gdayf.common.normalizationset import NormalizationSet
from gdayf.common.storagemetadata import StorageMetadata
from gdayf.metrics.metricmetadata import MetricMetadata
from gdayf.models.frameworkmetadata import FrameworkMetadata

## Define all objects, functions and structured related to Analysis_Results for one execution (final json structure)
# on OrderedDict format


## Class ArMetadata manage the Analysis results structe on OrderedDict format and exportable to json
class ArMetadata (OrderedDict):

    ## The constructor
    # Generate an empty AR class with all DayF objects elements initialized to correct types
    def __init__(self):
        OrderedDict.__init__(self)
        self['model_id'] = None
        self['version'] = None
        self['type'] = None
        self['objective_column'] = None
        self['timestamp'] = None
        self['load_path'] = StorageMetadata()
        self['metrics'] = MetricMetadata()
        self['normalizations_set'] = NormalizationSet()
        self['data_initial'] = DFMetada()
        self['data_normalized'] = DFMetada()
        self['model_parameters'] = FrameworkMetadata()
        self['ignored_parameters'] = None
        self['full_parameters_stack'] = None
        self['log_path'] = StorageMetadata()
        self['json_path'] = StorageMetadata()
        self['status'] = -1

    ## Get json format string associted to class OredredDict parameters with encoding utf-8 and indent = 4
    # @param self object pointer
    # @return OrderedDict() structure associated to json file with indent 4 and encoding utf-8
    def get_json(self):
        return dump(self, indent=4, enconding='utf8')

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
    print(m)

