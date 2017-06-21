#!/usr/bin/python3

from json import dumps, loads
from collections import OrderedDict
from gdayf.common.utils import dtypes
import operator


class DFMetada(OrderedDict):
    def __init__(self):
        OrderedDict.__init__(self)
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
            auxdict['cardinality'] = str(int(dataframe[col].value_counts().describe()['count']))
            hist = dataframe[col].value_counts().to_dict()
            auxdict['histogram'] = OrderedDict()
            for tupla in sorted(hist.items(), key=operator.itemgetter(0)):
                if int(auxdict['cardinality']) < 200:
                    auxdict['histogram'][str(tupla[0])] = str(tupla[1])
            auxdict['distribution'] = 'Not implemented yet'
            self['columns'].append(auxdict)
        self['correlation'] = dataframe.corr().to_dict()
        #return dumps(self, indent=4)
        return self

    def pop(self, key, default=None):
        return 1

    def popitem(self, last=True):
        return 1

if __name__ == "__main__":
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from pandas import concat
    import operator
    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")

    pd_train_dataset = concat([inputHandlerCSV().inputCSV(''.join(source_data) + "football.train2.csv"),
                               inputHandlerCSV().inputCSV(''.join(source_data) + "football.valid2.csv")],
                              axis=0)

    m = DFMetada()
    print(m.getDataFrameMetadata(pd_train_dataset, 'pandas'))

