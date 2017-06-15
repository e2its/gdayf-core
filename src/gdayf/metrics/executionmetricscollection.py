from collections import OrderedDict
##  Define comomn execution base structure as OrderedDict() of common datasets
#  on an unified way


# Class Base for Execution metricts as OrderedDict
#
class MetricCollection (OrderedDict):
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