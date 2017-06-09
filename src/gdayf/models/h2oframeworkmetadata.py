from collections import OrderedDict
from gdayf.conf.loadconfig import LoadConfig

class H2OFrameworkMetadata (OrderedDict):
    def __init__(self):
        super(H2OFrameworkMetadata, self).__init__()
        self._config = LoadConfig().get_config()
        self['h2o'] = self._config['frameworks']['h2o']['models']
