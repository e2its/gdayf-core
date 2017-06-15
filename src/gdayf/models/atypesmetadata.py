from collections import OrderedDict

##  Define Analysis Types for DayF product
#  on an unified way

## Generate OrderedDict() from analysis types accepted
# returning the structure to be added on ModelMetadata
class ATypesMetadata (list):

    @classmethod
    # Classmethod Define analysis types allowed
    # @param cls class pointer
    # @return analysis types allowed
    def get_artypes(cls):
        return ['binomial', 'multinomial', 'regression', 'topology']

    ## Constructor
    #
    # **Kargs [binomial=boolean, multinomial=boolean, regression=boolean, topology=boolean]
    def __init__(self, **kwargs):
        list().__init__(self)
        for pname, pvalue in kwargs.items():
            if pname in ATypesMetadata().get_artypes():
                artype = OrderedDict()
                artype['type'] = str(pname)
                artype['active'] = pvalue
                self.append(artype)


if __name__ == "__main__":
    m = ATypesMetadata(binomial=True, topology=True )
    print(m)