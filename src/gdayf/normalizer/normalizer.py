#!/usr/bin/python3
import numpy


class Normalizer:
    def __init__(self):
        True

    def normalizeDataFrame(self, dataframe, normalizemd):
        if (normalizemd['type'] == 'pandas'):
            for col in normalizemd['columns']:
                if (col['class'] == 'mean'):
                    self.normalizeMean(dataframe,
                                       col['name'],
                                       col['objective']['mean'],
                                       col['objective']['std'])
                elif (col['class'] == 'working_range'):
                    self.normalizeWorkingRange(dataframe,
                                               col['name'],
                                               col['objective']['minval'],
                                               col['objective']['maxval'])
                elif (col['class'] == 'discretize'):
                    self.normalizeDiscretize(dataframe,
                                             col['name'],
                                             col['objective']['buckets_number'],
                                             col['objective']['fixed_size'])
                elif (col['class'] == 'agregation'):
                    self.normalizeAgregation(dataframe,
                                             col['name'],
                                             col['objective']['bucket_radio'])
                elif (col['class'] == 'missing_values'):
                    self.normalizeMissingValues(dataframe,
                                                col['name'],
                                                col['objective']['mean'],
                                                col['objective']['mode'],
                                                col['objective']['fixed'],
                                                col['objective']['k_nearest'],
                                                col['objective']['value'])
                elif (col['class'] == 'binary_encoding'):
                    self.normalizeBinaryEncoding(dataframe,
                                                 col['name'],
                                                 col['objective']['minval'],
                                                 col['objective']['maxval'])
                else:
                    print("Nothing to Normalize")

    def normalizeMean(self, dataframe, col, mean=0, std=1):
        True

    def normalizeWorkingRange(self, dataframe, col, minval=0, maxval=1):
        assert(maxval > minval)
        if (dataframe[col].dtype != numpy.object):
            dataframe[col] = (maxval - minval) * ((dataframe[col] - dataframe[col].min()) / (dataframe[col].max() - dataframe[col].min())) + maxval
        return dataframe
