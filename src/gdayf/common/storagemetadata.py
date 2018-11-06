## @package gdayf.common.storagemetadata
# Define all objects, functions and structured related to adding storage information metadata (json structure)
# on list[OrderedDict] format

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
'''

from gdayf.common.utils import hash_key
from collections import OrderedDict
from os import path
from gdayf.common.utils import get_model_fw
from copy import deepcopy


## Class storage metadata
# format [{value: , fstype:['localfs', 'hdfs', 'mongoDB'], hash_value : "", hash_type:  ['MD5','SHA256']
class StorageMetadata (list):
    ## Constructor
    # return empty list of StoragMetadata
    def __init__(self, e_c):
        self._ec = e_c
        self._config = self._ec.config.get_config()
        #self._labels = self._ec.labels.get_config()['messages']['controller']
        list.__init__(self)

    ## class used to add storage locations to StorageMetadata. use list().append method to include correct media and
    # hash_value for file over OrderedDict() object
    # overriding list().append method
    # @param self object pointer location (optional)
    # @param value type full file path (string)
    # @param fstype in ['localfs', 'hdfs', 'mongoDB'] default value 'localfs'
    # @param hash_type in  ['MD5','SHA256'] , default value 'MD5'
    # @return None
    def append(self, value, fstype='localfs', hash_type='MD5'):
        try:
            assert fstype in ['localfs', 'hdfs', 'mongoDB']
            assert hash_type in ['MD5', 'SHA256']

            fs = OrderedDict()
            fs['type'] = fstype
            fs['value'] = value
            fs['hash_type'] = hash_type

            if fstype == 'localfs' and path.exists(value) and not path.isdir(value):
                fs['hash_value'] = hash_key(hash_type=hash_type, filename=fs['value'])
            else:
                fs['hash_value'] = None
            super(StorageMetadata, self).append(fs)
        except:
            super(StorageMetadata, self).append(value)

    ## method used to get relative load path from config.json
    # @param self object pointer location (optional)
    # @param include enable localfs
    # @return relative path string
    def get_load_path(self, include=False):
        return self.exclude_debug_fs(deepcopy(self._config['storage']['load_path']), include=include)

    ## method used to get relative log path from config.json
    # @param self object pointer location (optional)
    # @return relative path string
    def get_log_path(self):
        return deepcopy(self._config['storage']['log_path'])

    ## method used to get relative json path from config.json
    # @param self object pointer location (optional)
    # @param include enable localfs
    # @return relative path string
    def get_json_path(self, include=False):
        return self.exclude_debug_fs(deepcopy(self._config['storage']['json_path']), include=include)

    ## method used to get relative prediction path from config.json
    # @param self object pointer location (optional)
    # @param include enable localfs
    # @return relative path string
    def get_prediction_path(self, include=False):
        return self.exclude_debug_fs(deepcopy(self._config['storage']['prediction_path']), include=include)

    ## method used to exclude localfs in non-debug modes
    # @param storage_metadata StorageMetadata object
    # @param include enable localfs
    def exclude_debug_fs(self, storage_metadata, include=False):
        equals = list()
        if not self._config['storage']['localfs_debug_mode'] and not include:
            for each_storage in storage_metadata:
                if each_storage['type'] == 'localfs':
                    equals.append(each_storage)
            for deleter in equals:
                storage_metadata.remove(deleter)

        return storage_metadata

## Method to Generate json StorageMetadata for Armetadata
# @param e_c context pointer
# @param armetadata structure to be stored
# @param json_type ['persistence','json']
def generate_json_path(e_c, armetadata, json_type='json'):
    config = e_c.config.get_config()
    fw = get_model_fw(armetadata)

    model_id = armetadata['model_parameters'][fw]['parameters']['model_id']['value']
    compress = config['persistence']['compress_json']
    json_storage = StorageMetadata(e_c)

    command = 'json_storage.get_' + json_type + '_path()'
    for each_storage_type in eval(command):
        if each_storage_type['type'] in ['localfs', 'hdfs']:
            primary_path = config['storage'][each_storage_type['type']]['value']
            source_data = list()
            source_data.append(primary_path)
            source_data.append('/')
            source_data.append(armetadata['user_id'])
            source_data.append('/')
            source_data.append(armetadata['workflow_id'])
            source_data.append('/')
            source_data.append(armetadata['model_id'])
            source_data.append('/')
            source_data.append(fw)
            source_data.append('/')
            source_data.append(armetadata['type'])
            source_data.append('/')
            source_data.append(str(armetadata['timestamp']))
            source_data.append('/')

            specific_data = list()
            specific_data.append(each_storage_type['value'])
            specific_data.append('/')
            specific_data.append(model_id)
            specific_data.append('.json')
            if compress:
                specific_data.append('.gz')

            json_path = ''.join(source_data)
            json_path += ''.join(specific_data)
            json_storage.append(value=json_path, fstype=each_storage_type['type'],
                                hash_type=each_storage_type['hash_type'])

        else:
            if json_type == 'json':
                json_storage.append(value=armetadata['user_id'], fstype=each_storage_type['type'],
                                    hash_type=each_storage_type['hash_type'])
            else:
                json_storage.append(value=each_storage_type['value'], fstype=each_storage_type['type'],
                                    hash_type=each_storage_type['hash_type'])

    command = json_type + '_path'
    armetadata[command] = json_storage
