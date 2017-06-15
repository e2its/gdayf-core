## @package gdayf.models.h2omodelmetadata
# Define Base Model for H2OFramework
#  on an unified way. Base for all Models

from gdayf.models.modelmetadata import ModelMetadata


## Generate H2O Model base Class and initialize base members
class H2OModelMetadata(ModelMetadata):
    ## Constructor
    def __init__(self):
        ModelMetadata.__init__(self)
        # @var _config
        # Initialized _config to h2o all models default values
        self._config = self._config['h2o']

if __name__ == "__main__":
    m = H2OModelMetadata()
    print(m.get_default())