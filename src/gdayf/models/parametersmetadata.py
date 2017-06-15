## @package gdayf.models.parametersmetadata
# Define all objects, functions and structured related to manage Model Parameters
# Structure: OrderedDict():
# value :
# Seleccionable: boolean
# Type: Internal structure and restriction over value as reference
from collections import OrderedDict


##Class Parameter metadata
class ParameterMetadata(OrderedDict):
    ## Constructor
    def __init__(self, value, seleccionable=True, type=None):
        OrderedDict.__init__(self)
        self['value'] = value
        self['seleccionable'] = bool(seleccionable)
        self['type'] = str(type)

    ## Method for setting values on seleccionable field
    # @param boolean True/False
    def set_seleccionable (self, boolean):
        self['seleccionable'] = boolean

    ## Method for getting values on seleccionable field
    # @return  Seleccionable value
    def get_seleccionable(self):
        return self['seleccionable']

