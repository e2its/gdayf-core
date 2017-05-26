#!/usr/bin/python3

import pandas as pd
import json


class inputHandler:
    def __init__(self):
        True


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

    def createDataFramePandasMetadata(self, dataframe):
        md = {}
        md['type'] = 'pandas'
        md['rowcount'] = dataframe.shape[0] - 1
        md['cols'] = dataframe.shape[1]
        md['timeformat'] = ''
        md['columns'] = {}

        return md


if __name__ == "__main__":
    ih = inputHandlerCSVLocal()
    datos = ih.inputCSV('/home/luis/FL_insurance_sample.csv')
    print("Tipos de datos: %s" % datos.dtypes)
    metadatos = ih.createDataFramePandasMetadata(datos)
    print("Metadatos: %s" % metadatos)
