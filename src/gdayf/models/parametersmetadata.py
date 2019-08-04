## @package gdayf.models.parametersmetadata
# Define all objects, functions and structured related to manage Model Parameters
# Structure: OrderedDict():
# value :
# Seleccionable: boolean
# Type: Internal structure and restriction over value as reference

'''
 * This file is part of the gDayF AutoML Core Framework project
 * distribution (https://github.com/e2its/gdayf-core).
 * Copyright (c) 2016-2019 Jose Luis Sanchez del Coso <e2its.es@gmail.com>.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ** Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
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

