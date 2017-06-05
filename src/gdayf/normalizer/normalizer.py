#!/usr/bin/python3
import numpy


class Normalizer:
    def __init__(self):
        True

    def normalizeWorkingRange(self, dataframe, minval=0, maxval=1):
        assert(maxval > minval)
        for col in dataframe.columns:
            if (dataframe[col].dtype != numpy.object):
                dataframe[col] = (maxval - minval) * ((dataframe[col] - dataframe[col].min()) / (dataframe[col].max() - dataframe[col].min())) + maxval
        return dataframe
