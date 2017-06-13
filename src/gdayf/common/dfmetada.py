#!/usr/bin/python3

from json import dump
from collections import OrderedDict


class DFMetada(OrderedDict):
    def __init__(self):
        super().__init__()
        self['type'] = None
        self['rowcount'] = None
        self['cols'] = None
        self['timeformat'] = 'dd-mm-yyyy HH:mm:ss:ms'
        self['columns'] = list()

    def getDataFrameMetadata(self, dataframe, typedf):
        self['type'] = '%s' % typedf
        self['rowcount'] = dataframe.shape[0] - 1
        self['cols'] = dataframe.shape[1]
        self['timeformat'] = 'dd-mm-yyyy HH:mm:ss:ms'
        for col in dataframe.columns:
            summary = dataframe[col].describe()
            auxdict = OrderedDict()
            auxdict['name'] = dataframe[col].name
            auxdict['type'] = str(dataframe[col].dtype)
            for comp in ['min', 'max', 'mean', 'std', '25%', '50%', '75%']:
                try:
                    auxdict[comp] = str(summary[comp])
                except KeyError:
                    auxdict[comp] = 'NaN'
            auxdict['zeros'] = str(dataframe[col][dataframe[col] == 0].count())
            auxdict['missed'] = str(
                dataframe[col].isnull().values.ravel().sum())
            self['columns'].append(auxdict)

        return None
        #return dump(self, indent=4)

    def pop(self, key, default=None):
        return 1

    def popitem(self, last=True):
        return 1
