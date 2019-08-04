## @package gdayf.models.atypesmetadata
# Define Analysis Types for DayF product
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
