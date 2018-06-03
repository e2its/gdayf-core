## @package gdayf.handlers.inputhandler

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
'''

import pandas as pd
from gdayf.common.dfmetada import DFMetada

class inputHandler:
    def __init__(self):
        True


class inputHandlerCSV (inputHandler):
    def __init__(self):
        super(inputHandlerCSV, self).__init__()

    def inputCSV(self, filename=None, **kwargs):
        return pd.read_csv(filename, **kwargs)

    def getCSVMetadata(self, dataframe, typedef):
        return DFMetada.getDataFrameMetadata(dataframe, typedef)


if __name__ == "__main__":
    ih = inputHandlerCSV()

# Pruebas para cargar un fichero local csv
    datos = ih.inputCSV(
        '/home/luis/desarrollo/pruebas/FL_insurance_sample.csv')
    metadatas = ih.getCSVMetadata(datos, 'pandas')
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
    metadatas = ih.getCSVMetadata(datos, 'pandas')
    file = open('salida_local_aeropuerto.json', 'w')
    file.writelines(metadatas)
    file.close()

# Pruebas para cargar un fichero csv remoto mediante su url
    datos = ih.inputCSV(
        'http://catalogo.sevilla.org/dataset/621613fb-8315-4c00-915e-5ff0538f92c2/resource/21923b17-2c88-4b1f-9e50-b55cf2bbef6a/download/estadoejecuciongastos1semestregmu.csv')
    metadatas = ih.getCSVMetadata(datos, 'pandas')
    file = open('salida_remota.json', 'w')
    file.writelines(metadatas)
    file.close()
