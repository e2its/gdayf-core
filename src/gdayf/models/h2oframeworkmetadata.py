## @package gdayf.models.h2oframeworkmetadata
#  Define Base H2OFramework methods and members
#  on an unified way.
from gdayf.models.frameworkmetadata import FrameworkMetadata
import copy


## Generate Framework base Class and base members
# Initialize ['h2o'] to list()
class H2OFrameworkMetadata (FrameworkMetadata):
    ## Constructor
    def __init__(self):
        FrameworkMetadata.__init__(self)
        self._config = self._config['h2o']
        self['h2o'] = list()

    ## Method used to get a copy default config values for framework
    # @return a copy from structure
    def get_default(self):
        return copy.deepcopy(self._config)

    ## Method used to get default models values from framework
    # @return pointer to structure
    def load_default_models(self):
        self.get_default()['models']
        return self.get_default()['models']

    ## Method used to append models
    # @param self object pointer
    # @param modelmetadata Model Metadata subclass object [glm, gbm, depplearning, drf]
    # @return status (success o) (error 1)
    def append(self, modelmetadata):
        try:
            self['h2o'].append(modelmetadata)
            return 0
        except Exception:
            return 1

    ## Method used get a copy of [h2o] models
    # @param self object pointer
    def get(self):
        return copy.deepcopy(self['h2o'])

if __name__ == "__main__":
    m = H2OFrameworkMetadata()
    import json

    m.append(m.load_default_models()[0])
    print (json.dumps(m.get(), indent=4))
    '''print(m.get().items())
    print(m.load_default_models().items())
    print(json.dumps(m._config, indent=4))'''

