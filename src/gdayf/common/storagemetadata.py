## @package gdayf.common.storagemetadata
# Define all objects, functions and structured related to adding storage information metadata (json structure)
# on list[OrderedDict] format

from gdayf.common.utils import hash_key
from gdayf.conf.loadconfig import LoadConfig
from collections import OrderedDict
from os import path


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
        assert fstype in ['localfs', 'hdfs', 'mongoDB']
        assert hash_type in ['MD5', 'SHA256']

        fs = OrderedDict()
        fs['type'] = fstype
        fs['value'] = value
        fs['hash_type'] = hash_type

        if path.exists(value):
            fs['hash_value'] = hash_key(hash_type=hash_type, filename=fs['value'])
        else:
            fs['hash_value'] = None
        super(StorageMetadata, self).append(fs)

    ## method used to get realtive path from config.json
    # @param self object pointer location (optional)
    # @return relative path string
    def get_load_path(self):
        return LoadConfig().get_config()['storage']['load_path']

    ## method used to get realtive path from config.json
    # @param self object pointer location (optional)
    # @return relative path string
    def get_log_path(self):
        return LoadConfig().get_config()['storage']['log_path']

    ## method used to get realtive path from config.json
    # @param self object pointer location (optional)
    # @return relative path string
    def get_json_path(self):
        return LoadConfig().get_config()['storage']['json_path']


if __name__ == '__main__':
    print(get_load_path())
    print(get_log_path())
    print(get_json_path())