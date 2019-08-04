## @package gdayf.handlers.inputhandler

'''
 * This file is part of the gDayF AutoML Core Framework project
 * distribution (https://github.com/e2its/gdayf-core).
 * Copyright (c) 2016-2019 Jose Luis Sanchez del Coso <e2its.es@gmail.com>.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ** Written by Luis Sanchez Bejarano <e2its.es@gmail.com>, 2016-2019
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

