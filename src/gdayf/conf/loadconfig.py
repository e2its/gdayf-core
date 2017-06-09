from collections import OrderedDict
import json
from os import path
import copy


class LoadConfig(object):
    def __init__(self, configfile=r'D:\e2its-dayf.svn\gdayf\branches\0.0.3-team03\src\gdayf\conf\config.json'):
        self._config = None
        self._configfile = configfile
        if path.exists(configfile):
            with open(configfile, 'rt') as f:
                try: 
                    self._config = json.load(f, object_hook=OrderedDict, encoding='utf8')
                except Exception:
                    raise Exception
        else:
            raise Exception

    def get_config(self):
        return copy.deepcopy(self._config)

    def get_configfile(self):
        return self._configfile
