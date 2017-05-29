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


class inputHandlerCSV (inputHandler):
    def __init__(self):
        super(inputHandlerCSV, self).__init__()

    def inputCSV(self, filename=None, **kwargs):
        return pd.read_csv(filename, **kwargs)


if __name__ == "__main__":
    ih = inputHandlerCSV()

# Pruebas para cargar un fichero local csv
    datos = ih.inputCSV(
        '/home/luis/desarrollo/pruebas/FL_insurance_sample.csv')
    metadatas = ih.getDataFrameMetadata(datos, 'pandas')
    file = open('salida_local.json', 'w')
    file.writelines(metadatas)
    file.close()

# Pruebas para cargar un fichero local csv indicando cual es la cabecera,
# eligiendo las seis primeras columnas tomando con columna indice la 0 y
# con codificación iso-8859-1
    datos = ih.inputCSV(
        filename='/home/luis/desarrollo/pruebas/valoresclimatologicos_almeria-aeropuerto.csv',
        header=1,
        usecols=[0, 1, 2, 3, 4, 5],
        index_col=[0],
        encoding='iso-8859-1')
    metadatas = ih.getDataFrameMetadata(datos, 'pandas')
    file = open('salida_local_aeropuerto.json', 'w')
    file.writelines(metadatas)
    file.close()

# Pruebas para cargar un fichero csv remoto mediante su url
    datos = ih.inputCSV(
        'http://catalogo.sevilla.org/dataset/621613fb-8315-4c00-915e-5ff0538f92c2/resource/21923b17-2c88-4b1f-9e50-b55cf2bbef6a/download/estadoejecuciongastos1semestregmu.csv')
    metadatas = ih.getDataFrameMetadata(datos, 'pandas')
    file = open('salida_remota.json', 'w')
    file.writelines(metadatas)
    file.close()
