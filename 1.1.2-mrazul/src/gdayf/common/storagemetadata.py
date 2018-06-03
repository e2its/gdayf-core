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
from gdayf.conf.loadconfig import LoadConfig
from collections import OrderedDict
from os import path
from gdayf.common.utils import get_model_fw


## Class storage metadata
# format [{value: , fstype:['localfs', 'hdfs', 'mongoDB'], hash_value : "", hash_type:  ['MD5','SHA256']
class StorageMetadata (list):
    ## Constructor
    # return empty list of StoragMetadata
    def __init__(self,):
        list.__init__(self)

    ## class used to add storage locations to StorageMetadata. use list().append method to include correct media and
    # hash_value for file over OrderedDict() object
    # overriding list().append method
    # @param self object pointer location (optional)
    # @param value type full file path (string)
    # @param fstype in ['localfs', 'hdfs', 'mongoDB'] default value 'localfs'
    # @param hash_type in  ['MD5','SHA256'] default value 'MD5'
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
    # @return relative path string
    @staticmethod
    def get_load_path():
        return LoadConfig().get_config()['storage']['load_path']

    ## method used to get relative log path from config.json
    # @return relative path string
    @staticmethod
    def get_log_path():
        return LoadConfig().get_config()['storage']['log_path']

    ## method used to get relative json path from config.json
    # @param self object pointer location (optional)
    # @return relative path string
    @staticmethod
    def get_json_path():
        return LoadConfig().get_config()['storage']['json_path']


## Function to Generate json StorageMetadata for Armetadata
# @param armetadata structure to be stored

def generate_json_path(armetadata):
    config = LoadConfig().get_config()
    fw = get_model_fw(armetadata)

    model_id = armetadata['model_parameters'][fw]['parameters']['model_id']['value']
    compress = config['persistence']['compress_json']
    json_storage = StorageMetadata()
    for each_storage_type in json_storage.get_json_path():
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
            json_storage.append(value=armetadata['user_id'], fstype=each_storage_type['type'],
                                hash_type=each_storage_type['hash_type'])

    armetadata['json_path'] = json_storage