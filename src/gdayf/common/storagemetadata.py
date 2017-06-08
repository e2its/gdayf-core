from collections import OrderedDict
from gdayf.common.utils import hash_key
from os import path


class StorageMetadata (list):

    def __init__(self):
        super(StorageMetadata, self).__init__()

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




