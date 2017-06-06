#!/usr/bin/python3
import numpy


class Normalizer:
    def __init__(self):
        True

    def normalizeDataFrame(self, dataframe, normalizemd):
        if (normalizemd['type'] == 'pandas'):
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
                elif (col['class'] == 'agregation'):
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
                    dataframe[col['name']] = self.normalizeBinaryEncoding(dataframe[col['name']],
                                                 col['objective']['minval'],
                                                 col['objective']['maxval'])
                else:
                    print("Nothing to Normalize")

                return dataframe

    def normalizeMean(self, dataframe, mean=0, std=1):
        True

    def normalizeWorkingRange(self, dataframe, minval=0, maxval=1):
        assert(maxval > minval)
        if (dataframe.dtype != numpy.object):
            dataframe = (maxval - minval) * ((dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())) + maxval
        return dataframe
