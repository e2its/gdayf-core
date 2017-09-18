## @package gdayf.common.armetadata
# Define all objects, functions and structured related to Analysis_Results for one execution (final json structure)
# on OrderedDict format

from collections import OrderedDict
from copy import deepcopy
from json import dumps
from gdayf.common.dfmetada import DFMetada
from gdayf.common.normalizationset import NormalizationSet
from gdayf.common.storagemetadata import StorageMetadata
from gdayf.metrics.metricmetadata import MetricMetadata
from gdayf.metrics.executionmetriccollection import ExecutionMetricCollection
from gdayf.models.frameworkmetadata import FrameworkMetadata
from gdayf.common.utils import get_model_fw


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
        self['round'] = 1
        self['execution_seconds'] = 0
        self['load_path'] = None
        self['metrics'] = OrderedDict()
        self['normalizations_set'] = None
        self['dataset'] = None
        self['dataset_hash_value'] = None
        self['data_initial'] = None
        self['data_normalized'] = None
        self['model_parameters'] = None
        self['ignored_parameters'] = None
        self['full_parameters_stack'] = None
        self['log_path'] = None
        self['json_path'] = None
        self['predecessor'] = None
        self['status'] = -1

    ## Get json format string associted to class OredredDict parameters with encoding utf-8 and indent = 4
    # @param self object pointer
    # @return OrderedDict() structure associated to json file with indent 4 and encoding utf-8
    def get_json(self):
        return dumps(self, indent=4)
    
    ## Get ArMetadata and make a base copy of main parameters to get an ArMetadata structure to be analyzed
    # @param self object pointer
    # @param deepness of analysis
    # @return OrderedDict() structure associated to json file with indent 4 and encoding utf-8
    def copy_template(self, deepness=0):
        new_model = ArMetadata()
        new_model['model_id'] = deepcopy(self['model_id'])
        new_model['version'] = deepcopy(self['version'])
        new_model['type'] = deepcopy(self['type'])
        new_model['objective_column'] = deepcopy(self['objective_column'])
        new_model['timestamp'] = deepcopy(self['timestamp'])
        new_model['round'] = deepness
        new_model['execution_seconds'] = 0.0
        new_model['tolerance'] = 0.0
        new_model['predecessor'] = self['model_parameters'][get_model_fw(self)]['parameters']['model_id']['value']
        new_model['normalizations_set'] = deepcopy(self['normalizations_set'])
        new_model['dataset'] = deepcopy(self['dataset'])
        new_model['dataset_hash_value'] = deepcopy(self['dataset_hash_value'])
        new_model['data_initial'] = deepcopy(self['data_initial'])
        new_model['data_normalized'] = deepcopy(self['data_normalized'])
        new_model['model_parameters'] = deepcopy(self['model_parameters'])
        new_model['ignored_parameters'] = deepcopy(self['ignored_parameters'])
        return new_model

    ## Anulate pop fetures from OrderedDict parent class.
    # Stability  proposals
    def pop(self, key, default=None):
        return 1

    ## Anulate popitem fetures from OrderedDict parent class
    # Stability  proposals
    def popitem(self, last=True):
        return 1


## Package Function getting ArMetadata and make a base copy of parameters to get an ArMetadata structure base to predict
# @param armetadata structure
# @return ArMetadata() structure OrderedDict and json compatible
def deep_ordered_copy(armetadata):
    new_model = ArMetadata()
    for key, value in armetadata.items():
        if key not in ['load_path', 'json_path', 'load_path' ]:
            new_model[key] = deepcopy(value)
        else:
            if value is not None:
                new_model[key] = list()
                for each_storage in armetadata[key]:
                    new_model[key].append(OrderedDict(
                        {'value': each_storage['value'],
                         'type': each_storage['type'],
                         'hash_value': each_storage['hash_value'],
                         'hash_type': each_storage['hash_type']
                         }
                    ))
            else:
                new_model[key] = None


    return new_model


'''## Main block only for testing issues
if __name__ == "__main__":
    ## Varible for testinf propouses
    m = ArMetadata()
    print(m.get_json())'''

