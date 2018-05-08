## @package gdayf.models.atypesmetadata
# Define Analysis Types for DayF product
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

## Generate OrderedDict() from analysis types accepted
# returning the structure to be added on ModelMetadata
class ATypesMetadata (list):

    @classmethod
    ## Classmethod Define analysis types allowed
    # @param cls class pointer
    # @return analysis types allowed
    def get_artypes(cls):
        return ['binomial', 'multinomial', 'regression', 'topology', 'anomalies', 'clustering']

    ## Constructor
    #
    # @param **kwargs [binomial=boolean, multinomial=boolean, regression=boolean, topology=boolean]
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
