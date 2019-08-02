## @package gdayf.models.modelmetadata
# Define Base Model methods and members
#  on an unified way. Base for all Models

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

import copy
from gdayf.conf.loadconfig import LoadConfig
from collections import OrderedDict


## Generate Model base Class and base members
class ModelMetadata(object):
    ## Constructor
    def __init__(self, e_c):
        # @var _config
        # Load default parameters for Models
        self._ec = e_c
        self._config = self._ec.config.get_config()['frameworks']
        # @var model
        #  Initialized model to None
        self.model = OrderedDict()
    ## Method use to get the model
    def get_model(self):
        return self.model

    ## Method use to get the config on OrderedDict() format
    def get_default(self):
        return copy.deepcopy(self._config)

    ## Method use to get a copy of model
    # @return deep copy of model
    def get_copy_model(self):
        return copy.deepcopy(self.model)




'''if __name__ == "__main__":
    m = ModelMetadata()
    print(m._config['h2o']['models'])'''

