## @package gdayf.handlers.inputhandler

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Luis Sanchez Bejarano <e2its.es@gmail.com>, 2016-2019
'''

from pandas import read_csv as read_csv
from gdayf.common.dfmetada import DFMetada

class inputHandler:
    def __init__(self):
        pass

class inputHandlerCSV (inputHandler):
    ## The constructor
    def __init__(self):
        super(inputHandlerCSV, self).__init__()

    def inputCSV(self, filename=None, **kwargs):
        return read_csv(filename, **kwargs)

    def getCSVMetadata(self, dataframe, typedef):
        return DFMetada.getDataFrameMetadata(dataframe, typedef)

