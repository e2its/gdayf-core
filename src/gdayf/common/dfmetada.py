## @package gdayf.common.dfmetada
# Define all objects, functions and structured related to Data Analysis of input data
# on OrderedDict format

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by e2its <e2its.es@gmail.com>, 2016-2018
'''

from json import dumps
from collections import OrderedDict
from gdayf.common.constants import DTYPES
from gdayf.conf.loadconfig import LoadConfig
from copy import deepcopy
from pandas import cut
from hashlib import md5 as md5
from numpy import isnan
import operator


## Class DFMetadata manage the Data Analysis results structs on OrderedDict format and exportable to json
class DFMetada(OrderedDict):

    ## The constructor
    # Generate an empty DFMetada class with all elements initialized to correct types
    def __init__(self):
        OrderedDict.__init__(self)
        self._config = LoadConfig().get_config()["dfmetadata"]
        self['type'] = None
        self['rowcount'] = None
        self['cols'] = None
        self['timeformat'] = None
        self['columns'] = list()

    ## Get dataframe on pandas format and return and equivalent DFmetadata object
    # @param self object pointer
    # @param dataframe pandas type Dataframe
    # @param typedf Dataframe type (for registry use)
    # @return self object pointer
    def getDataFrameMetadata(self, dataframe, typedf):
        self['type'] = '%s' % typedf
        self['rowcount'] = dataframe.shape[0] - 1
        self['cols'] = dataframe.shape[1]
        self['timeformat'] = None
        for col in dataframe.columns:
            summary = dataframe[col].describe()
            auxdict = OrderedDict()
            auxdict['name'] = dataframe[col].name
            auxdict['type'] = str(dataframe[col].dtype)
            for comp in ['min', 'max', 'mean', 'std', '25%', '50%', '75%']:
                try:
                    auxdict[comp] = float(summary[comp])
                except KeyError:
                    auxdict[comp] = None
            if auxdict['type'] in DTYPES:
                 auxdict['zeros'] = float(dataframe[dataframe.loc[:, col] == 0][col].count())
            else:
                auxdict['zeros'] = None
            auxdict['missed'] = float(dataframe[col].isnull().values.ravel().sum())
            auxdict['cardinality'] = float(dataframe.loc[:, col].value_counts().describe()['count'])
            auxdict['histogram'] = OrderedDict()
            cardinality_limit = self._config["cardinality_limit"]
            if int(auxdict['cardinality']) <= cardinality_limit:
                hist = dataframe.loc[:, col].value_counts().to_dict()
                for tupla in sorted(hist.items(), key=operator.itemgetter(0)):
                    auxdict['histogram'][str(tupla[0])] = float(tupla[1])
                del hist
            else:
                try:
                    hist = cut(dataframe.loc[:, col], cardinality_limit).value_counts().to_dict()
                    for tupla in sorted(hist.items(), key=operator.itemgetter(0)):
                        auxdict['histogram'][str(tupla[0])] = float(tupla[1])
                    del hist
                except TypeError:
                    auxHist = dataframe[col].value_counts()
                    auxdict['histogram']['max'] = float(auxHist.max())
                    auxdict['histogram']['min'] = float(auxHist.min())
                    auxdict['histogram']['mean'] = float(auxHist.mean())
                    auxdict['histogram']['std'] = float(auxHist.std())
                    del auxHist
                except ValueError:
                    auxHist = dataframe[col].value_counts()
                    auxdict['histogram']['max'] = float(auxHist.max())
                    auxdict['histogram']['min'] = float(auxHist.min())
                    auxdict['histogram']['mean'] = float(auxHist.mean())
                    auxdict['histogram']['std'] = float(auxHist.std())
                    del auxHist

            auxdict['distribution'] = 'Not implemented yet'
            self['columns'].append(auxdict)
        self['correlation'] = dataframe.corr().to_dict()
        for key, value in deepcopy(self['correlation']).items():
            for subkey, subvalue in value.items():
                if (self._config['correlation_threshold'] >= subvalue >= -self._config['correlation_threshold']) \
                        or key == subkey or isnan(subvalue):
                    self['correlation'][key].pop(subkey)
        return self

    def pop(self, key, default=None):
        return 1

    def popitem(self, last=True):
        return 1

## Function oriented compare two dicts based on hash_key(json transformations)
# @param dict1
# @param dict2
# @return True if equals false in other case
def compare_dict(dict1, dict2):
    if dict1 is None or dict2 is None:
        return dict1 is None and dict2 is None
    else:
        ddict1 = dumps(OrderedDict(dict1))
        ddict2 = dumps(OrderedDict(dict2))
        #print( md5(ddict1.encode('utf-8')))
        #print( md5(ddict2.encode('utf-8')))
        return md5(ddict1.encode('utf-8')) == md5(ddict2.encode('utf-8'))

if __name__ == "__main__":
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from pandas import concat
    import operator
    source_data = list()
    source_data.append("/Data/Data/datasheets/Multinomial/ARM/")
    source_data.append("ARM-Metric-test.csv")

    pd_train_dataset = inputHandlerCSV().inputCSV(''.join(source_data))

    m = DFMetada()
    print(OrderedDict(m.getDataFrameMetadata(pd_train_dataset, 'pandas')))
    print(dumps(m.getDataFrameMetadata(pd_train_dataset, 'pandas'), indent=4))

