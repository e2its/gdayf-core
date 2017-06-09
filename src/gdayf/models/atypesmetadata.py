from collections import OrderedDict


class ATypesMetadata (list):

    @classmethod
    def get_artypes(cls):
        return ['binomial', 'multinomial', 'regression', 'topology']

    def __init__(self, **kwargs):
        super(ATypesMetadata, self).__init__(self)
        for pname, pvalue in kwargs.items():
            if pname in ATypesMetadata().get_artypes():
                artype = OrderedDict()
                artype['type'] = str(pname)
                artype['active'] = pvalue
                self.append(artype)


