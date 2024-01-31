## @package gdayf.conf.loadconfig
# Define all objects, functions and structs related to load on system all configuration parameter from config.json

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

from collections import OrderedDict
import json
from os import path, makedirs
from shutil import copyfile


## Class Getting the config file place on default location and load all parameters on an internal variables
# named self._config on OrderedDict() format
class LoadConfig(object):
    _config = None
    _configfile = None

    ## Constructor
    def __init__(self, user_id='PoC_gDayF'):
        # @var _config protected member variable to store config parameters
        self._config = None
        configpath = path.join(path.dirname(__file__), '../../../.config')
        configfile = path.join(configpath, 'config.json')
        user_configpath = path.join(configpath, user_id)
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
            with open(self._configfile, 'rt') as f:
                try: 
                    # Desuppported (Encoding) version 3.9 self._config = json.load(f, object_hook=OrderedDict, encoding='utf8')
                    self._config = json.load(f, object_hook=OrderedDict)
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
                    self._config = json.load(f, object_hook=OrderedDict)[lang]
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

