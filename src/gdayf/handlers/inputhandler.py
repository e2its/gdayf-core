#!/usr/bin/python3

import pandas as pd
import json
from collections import OrderedDict


class inputHandler:
    def __init__(self):
        True

    def getDataFrameMetadata(self, dataframe, typedf):
        md = OrderedDict()
        md['type'] = '%s' % typedf
        md['rowcount'] = dataframe.shape[0] - 1
        md['cols'] = dataframe.shape[1]
        md['timeformat'] = ''
        columnlist = []
        for col in dataframe.columns:
            summary = dataframe[col].describe()
            print('%s' % summary)
            auxdict = OrderedDict()
            auxdict['name'] = dataframe[col].name
            auxdict['type'] = str(dataframe[col].dtype)
            auxdict['minval'] = str(summary['min'])
            auxdict['maxval'] = str(summary['max'])
            auxdict['mean'] = str(summary['mean'])
            columnlist.append(auxdict)
        md['columns'] = columnlist
        return json.dumps(md, indent=4)


class inputHandlerCSVLocal (inputHandler):
    def __init__(self):
        self.loadCSVDefaultParameters()

    def loadCSVDefaultParameters(self):
        self.sep = ','
        self.delimiter = None
        self.header = 'infer'
        self.index_col = None

    def inputCSV(self, filename=None, **kwargs):
        return pd.read_csv(filename)


if __name__ == "__main__":
    ih = inputHandlerCSVLocal()
    datos = ih.inputCSV(
        '/home/luis/desarrollo/pruebas/FL_insurance_sample.csv')
    print("Tipos de datos: %s" % datos.dtypes)
    metadatas = ih.getDataFrameMetadata(datos, 'pandas')
    file = open('salida.json', 'w')
    file.writelines(metadatas)
    file.close()
