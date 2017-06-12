#!/usr/bin/python3
import numpy as np
import pandas as pd
from sklearn import preprocesing

class Normalizer:
    def __init__(self):
        True

    def normalizeDataFrame(self, dataframe, normalizemd):
        if (normalizemd['type'] == 'pandas'):
            for col in normalizemd['columns']:
                if (col['class'] == 'mean'):
                    self.normalizeMean(dataframe[col['name']],
                                            col['objective']['mean'],
                                            col['objective']['std'])
                elif (col['class'] == 'working_range'):
                    self.normalizeWorkingRange(dataframe[col['name']],
                                            col['objective']['minval'],
                                            col['objective']['maxval'])
                elif (col['class'] == 'discretize'):
                    self.normalizeDiscretize(dataframe[col['name']],
                                            col['objective']['buckets_number'],
                                            col['objective']['fixed_size'])
                elif (col['class'] == 'agregation'):
                    self.normalizeAgregation(dataframe[col['name']],
                                             col['objective']['bucket_radio'])
                elif (col['class'] == 'missing_values'):
                    self.normalizeMissingValues(dataframe[col['name']],
                                            col['objective']['mean'],
                                            col['objective']['mode'],
                                            col['objective']['fixed'],
                                            col['objective']['k_nearest'],
                                            col['objective']['value'])
                elif (col['class'] == 'binary_encoding'):
                    self.normalizeBinaryEncoding(dataframe[col['name']],
                                            col['objective']['minval'],
                                            col['objective']['maxval'])
                else:
                    print("Nothing to Normalize")

    def normalizeMean(self, dataframe, mean=0, std=1):
        if (dataframe.dtype != numpy.object):
            #dataframe = (dataframe - dataframe.mean()) / ((dataframe.max()) - (dataframe.min()))
            #Aplico fórmula de estandarización:
            #dataframe = (dataframe - mean ) / std
            dataframe = preprocesing.scale(dataframe)

    def normalizeWorkingRange(self, dataframe, minval=0, maxval=1):
        assert(maxval > minval)
        if (dataframe.dtype != numpy.object):
            dataframe = (maxval - minval) * ((dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())) + maxval

    def normalizeDiscretize(self, dataframe, buckets_number, fixed_size):
        #Un número de buckets de tamaño fixed_size
        if (fixed_size == 0):
            True
        else:
            buckets = np.linspace(0,fixed_size,fixed_size*buckets_number)
            dataframe = np.digitize(dataframe,buckets)

    def normalizeAgregation(self, dataframe, bucket_radio):
        True

    def normalizeMissingValues(self, dataframe, mean, mode, fixed, k_nearest, value):
        #Aplicar media, moda o fijo
        if (mean == True):
            #dataframe = pd.DataFrame.fillna()
            True
        elif (mode == True):
            True
        elif(fixed == True):
            True
        else:
            print("Nothing to Normalize")

    def normalizeBinaryEncodig(self, dataframe, minval, maxval):
        True