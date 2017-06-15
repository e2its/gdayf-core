from collections import OrderedDict

class ParameterMetadata(OrderedDict):

    def __init__(self, value, seleccionable=True, type=None):
        OrderedDict.__init__(self)
        self['value'] = value
        self['seleccionable'] = bool(seleccionable)
        self['type'] = str(type)

    def set_seleccionable (self, boolean):
        self['seleccionable'] = boolean

    def get_seleccionable(self):
        return self['seleccionable']

