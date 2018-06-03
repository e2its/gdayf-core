## @package gdayf.metrics.executionmetriccollection
# Define common execution base structure as OrderedDict() of common datasets
#  on an unified way

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