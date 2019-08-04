## @package gdayf.metrics.executionmetriccollection
# Define common execution base structure as OrderedDict() of common datasets
#  on an unified way

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

# Class Base for Execution metricts as OrderedDict
#
class ExecutionMetricCollection (OrderedDict):
    ## Constructor initialize all to None
    # parama: self object pointer
    def __init__(self):
        OrderedDict.__init__(self)
        self['train'] = None
        self['valid'] = None
        self['xval'] = None
        self['predict'] = None

    ## Override OrderedDict().pop to do nothing
    def pop(self, key, default=None):
        return 1

    ## Override OrderedDict().popitem to do nothing
    def popitem(self, last=True):
        return 1