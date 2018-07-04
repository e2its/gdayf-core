## @package gdayf.models.frameworkmetadata
# Define Base Framework methods and members
#  on an unified way.

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
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