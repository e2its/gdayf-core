## @package gdayf.conf.loadconfig 
# Define all objects, functions and structs related to load on system all configuration parameter from config.json

from collections import OrderedDict
import json
from os import path


## Class Getting the config file place on default location and load all parameters on an internal variables
# named self._config on OrderedDict() format
class LoadConfig(object):
    _config = None
    _configfile = None

    ## Constructor
    def __init__(self, configfile=r'D:\e2its-dayf.svn\gdayf\branches\0.0.4-mrazul\src\gdayf\conf\config.json'):
        # @var _config protected member variable to store config parameters
        self._config = None
        # @var _configfile protected member variable to store configfile path
        self._configfile = configfile

        if path.exists(configfile):
            with open(configfile, 'rt') as f:
                try: 
                    self._config = json.load(f, object_hook=OrderedDict, encoding='utf8')
                except Exception:
                    raise Exception
        else:
            raise Exception

    ## Returns OrderedDict with all system configuration
    # @param self object pointer
    # @return OrderedDict() config
    def get_config(self):
        return self._config

    ## Returns configfile path
    # @param self object pointer
    # @return file full string path
    def get_configfile(self):
        return self._configfile

if __name__ == "__main__":
    m = LoadConfig(__name__)
    print(m.get_config()['logging'])
