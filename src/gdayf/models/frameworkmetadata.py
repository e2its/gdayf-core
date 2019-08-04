## @package gdayf.models.frameworkmetadata
# Define Base Framework methods and members
#  on an unified way.

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
 ** Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

from collections import OrderedDict
from gdayf.conf.loadconfig import LoadConfig


## Generate Framework base Class and base members
class FrameworkMetadata (OrderedDict):
    ## Constructor
    def __init__(self, e_c):
        # Load default parameters for Models as OrderedDict
        self._ec = e_c
        config = self._ec.config.get_config()['frameworks']
        for key, value in config.items():
            self[key] = value


if __name__ == "__main__":
    m = FrameworkMetadata()
    print (m['h2o'])