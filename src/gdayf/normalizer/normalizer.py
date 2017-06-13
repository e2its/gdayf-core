#!/usr/bin/python3
import numpy as np
import pandas as pd


class Normalizer:
    def __init__(self):
        True

    def normalizeDataFrame(self, df, normalizemd):
        if (normalizemd['type'] == 'pandas'):
            dataframe = df.copy()
            for col in normalizemd['columns']:
                if (col['class'] == 'mean'):
                    dataframe[col['name']] = self.normalizeMean(dataframe[col['name']],
                                                                col['objective']['mean'],
                                                                col['objective']['std'])
                elif (col['class'] == 'working_range'):
                    dataframe[col['name']] = self.normalizeWorkingRange(dataframe[col['name']],
                                               col['objective']['minval'],
                                               col['objective']['maxval'])
                elif (col['class'] == 'discretize'):
                    dataframe[col['name']] = self.normalizeDiscretize(dataframe[col['name']],
                                                                      col['objective']['buckets_number'],
                                                                      col['objective']['fixed_size'])
                elif (col['class'] == 'aggregation'):
                    dataframe[col['name']] = self.normalizeAgregation(dataframe[col['name']],
                                                                      col['objective']['bucket_radio'])
                elif (col['class'] == 'missing_values'):
                    dataframe[col['name']] = self.normalizeMissingValues(dataframe[col['name']],
                                                                         col['objective']['mean'],
                                                                         col['objective']['mode'],
                                                                         col['objective']['fixed'],
                                                                         col['objective']['k_nearest'],
                                                                         col['objective']['value'])
                elif (col['class'] == 'binary_encoding'):
                    dataframe[col['name']] = self.normalizeBinaryEncoding(dataframe[col['name']])
                else:
                    print("Nothing to Normalize")

            return dataframe

    def normalizeWorkingRange(self, dataframe, minval=0, maxval=1):
        assert(maxval > minval)
        if (dataframe[dataframe.columns[0]].dtype != np.object):
            dataframe = (maxval - minval) * ((dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())) + maxval

    def normalizeAggregation(self, dataframe, br=0.25):
        if (dataframe[dataframe.columns[0]].dtype != np.object):
            buckets = int(1 / br)
            q, bins = pd.qcut(dataframe.iloc[:], buckets, retbins=True)
            if (dataframe[dataframe.columns[0]].dtype != np.int):
                dataframe[dataframe <= bins[1]] = np.int(dataframe[dataframe <= bins[1]].mean())
            else:
                dataframe[dataframe <= bins[1]] = dataframe[dataframe <= bins[1]].mean()

    def normalizeBinaryEncoding(self, dataframe):
        return True
