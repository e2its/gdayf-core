## @package gdayf.conf.loadconfig
# Define all objects, functions and structs related to load on system all configuration parameter from config.json

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
import json
from os import path, makedirs
from shutil import copyfile
from  gdayf.core import global_var


## Class Getting the config file place on default location and load all parameters on an internal variables
# named self._config on OrderedDict() format
class LoadConfig(object):
    _config = None
    _configfile = None

    ## Constructor
    def __init__(self):
        # @var _config protected member variable to store config parameters
        self._config = None
        configpath = path.join(path.dirname(__file__), '../../../.config')
        configfile = path.join(configpath, 'config.json')
        user_configpath = path.join(configpath, global_var.get_id_user())
        user_configfile = path.join(user_configpath, 'config.json')
        if not path.exists(user_configfile):
            try:
                makedirs(user_configpath, mode=0o755, exist_ok=True)
                copyfile(configfile, user_configfile)
            except IOError:
                raise IOError

        # @var _configfile protected member variable to store configfile path
        self._configfile = user_configfile
        if path.exists(self._configfile):
            with open(configfile, 'rt') as f:
                try: 
                    self._config = json.load(f, object_hook=OrderedDict, encoding='utf8')
                except IOError:
                    raise IOError
        else:
            raise IOError

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



## Class Getting the config file place on default location and load all labels
# named self._config on OrderedDict() format
class LoadLabels(object):
    _config = None
    _configfile = None

    ## Constructor
    def __init__(self, lang='en'):
        # @var _config protected member variable to store config parameters
        self._config = None
        configpath = path.join(path.dirname(__file__), '../../../.config')
        configfile = path.join(configpath, 'labels.json')

        # @var _configfile protected member variable to store configfile path
        self._configfile = configfile
        if path.exists(self._configfile):
            with open(configfile, 'rt') as f:
                try:
                    self._config = json.load(f, object_hook=OrderedDict, encoding='utf8')[lang]
                except IOError:
                    raise IOError
        else:
            raise IOError

    ## Returns OrderedDict with all system labels
    # @param self object pointer
    # @return OrderedDict() config
    def get_config(self):
        return self._config

    ## Returns labelsfile path
    # @param self object pointer
    # @return file full string path
    def get_configfile(self):
        return self._configfile
