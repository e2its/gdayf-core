#!/usr/bin/python3

import json
from collections import OrderedDict


class DFMetada:
    def __init__(self):
        True

    def getDataFrameMetadata(self, dataframe, typedf):
        md = OrderedDict()
        md['type'] = '%s' % typedf
        md['rowcount'] = dataframe.shape[0] - 1
        md['cols'] = dataframe.shape[1]
        md['timeformat'] = 'dd-mm-yyyy HH:mm:ss:ms'
        columnlist = []
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
            columnlist.append(auxdict)
        md['columns'] = columnlist
        return json.dumps(md, indent=4)
