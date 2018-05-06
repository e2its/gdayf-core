## @package gdayf.models.sparkframeworkmetadata
#  Define Base sparkFramework methods and members
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

from gdayf.models.frameworkmetadata import FrameworkMetadata
from collections import OrderedDict
import copy


## Generate Framework base Class and base members
# Initialize ['spark'] to list()
class sparkFrameworkMetadata (object):
    ## Constructor
    def __init__(self, frameworks):
        # @var config
        # @var default_models
        # @var models
        self.default_models = frameworks['spark']['models']
        self.config = frameworks['spark']['conf']
        self.models = list()


    ## Method used to get a copy default config values for framework
    # @return a copy from structure
    def get_default(self):
        return copy.deepcopy(self.default_models)

    ## Method used to get default models values from framework
    # @return pointer to structure
    def load_default_models(self):
        self.models = self.get_default()

    ## Method used to append models
    # @param self object pointer
    # @param modelmetadata Model Metadata subclass object [glm, gbm, deeplearning, drf]
    # @return status (success o) (error 1)
    def append(self, modelmetadata):
        try:
            self.models.append(modelmetadata)
            return 0
        except Exception:
            return 1

    ## Method used get a copy of [h2o] models
    # @param self object pointer
    def get_models(self):
        return copy.deepcopy(self.models)

    ## Method used get a copy of [h2o] config
    # @param self object pointer
    def get_config(self):
        return copy.deepcopy(self.config)

if __name__ == "__main__":
    m = sparkFrameworkMetadata(FrameworkMetadata())
    import json

    m.append(m.get_default()[0])
    print(json.dumps(m.get_models(), indent=4))
    '''print(m.get_default())
    m.load_default_models()
    print(json.dumps(m.get_models(), indent=4))
    print(json.dumps(m.get_config(), indent=4))'''

