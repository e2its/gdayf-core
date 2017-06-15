import copy
from gdayf.conf.loadconfig import LoadConfig

##  Define Base Model methods and members
#  on an unified way. Base for all Models

## Generate Model base Class and base members


class ModelMetadata(object):
    ## Constructor
    def __init__(self):
        # @var _config
        # Load default parameters for Models
        self._config = LoadConfig().get_config()['frameworks']
        # @var model
        #  Initialized model to None
        self.model = None
    ## Method use to get the model
    def get_model(self):
        return self.model

    ## Method use to get the config on OrderedDict() format
    def get_default(self):
        return copy.deepcopy(self._config)

    ## Method use to get a copy of model
    # @return deep cpy of model
    def get_copy_model(self):
        return copy.deepcopy(self.model)

'''if __name__ == "__main__":
    m = ModelMetadata()
    print(m._config['h2o']['models'])'''

