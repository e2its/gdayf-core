from collections import OrderedDict


class MetricCollection (OrderedDict):
    def __init__(self):
        super(MetricCollection, self).__init__()
        self['train'] = None
        self['valid'] = None
        self['xval'] = None
        self['predict'] = None

    def pop(self, key, default=None):
        return 1

    def popitem(self, last=True):
        return 1