## @package gdayf.models.sparkframeworkmetadata
#  Define Base sparkFramework methods and members
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

