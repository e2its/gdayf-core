from collections import OrderedDict
from gdayf.common.utils import hash_key


class StorageMetadata (list):
    def __init__(self, value, fstype='fslocal', hash_='MD5'):
        assert fstype in ['fslocal', 'hdfs', 'mongoDB']
        assert fstype in ['MD5', 'SHA256']
        super().__init__()

        fs = OrderedDict()
        fs['type'] = fstype
        fs['value'] = value
        hash_ = OrderedDict()
        hash_['type'] = hash_
        try:
            hash_['value'] = hash_key(hash_['type'], fs['value'])
        except IOError:
            hash_['value'] = None
        fs['hash_list'] = list()
        fs['hash_list'].append(hash_)

        super().append(fs)

    def append(self, value, fstype='fslocal', hash_='MD5'):
        assert fstype in ['fslocal', 'hdfs', 'mongoDB']
        assert fstype in ['MD5', 'SHA256']

        fs = OrderedDict()
        fs['type'] = fstype
        fs['value'] = value
        hash_ = OrderedDict()
        hash_['type'] = hash_
        try:
            hash_['value'] = hash_key(hash_['type'], fs['value'])
        except IOError:
            hash_['value'] = None
        fs['hash_list'] = list()
        fs['hash_list'].append(hash_)
        super().append(fs)




