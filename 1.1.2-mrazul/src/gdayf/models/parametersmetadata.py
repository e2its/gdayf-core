## @package gdayf.models.parametersmetadata
# Define all objects, functions and structured related to manage Model Parameters
# Structure: OrderedDict():
# value :
# Seleccionable: boolean
# Type: Internal structure and restriction over value as reference

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


##Class Parameter metadata
class ParameterMetadata(OrderedDict):
    ## Constructor
    def __init__(self):
        OrderedDict.__init__(self)

    ## Method for setting values on all field
    # @param value value
    # @param seleccionable True/False
    # @param type value's options
    def set_value(self, value, seleccionable=True, type=None):
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

